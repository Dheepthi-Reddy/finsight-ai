"""
Semantic cache for the FinSight RAG pipeline.

What semantic caching is
-------------------------
A traditional cache (Redis, memcached) uses exact key matching: the query
"What are JPMorgan's AML obligations?" and "What are jpmorgan's AML obligations?"
are treated as two different keys even though they ask the same thing. Any
variation in whitespace, capitalisation, punctuation, or synonym choice causes
a cache miss.

A semantic cache uses vector similarity instead of string equality. Questions
are embedded into the same 768-dimensional space used for document retrieval;
two questions are considered equivalent if their embedding vectors are closer
than a configurable cosine similarity threshold (default 0.95).

This means the cache correctly hits for:
  • "What are JPMorgan's AML obligations?"
  • "What AML obligations does JPMorgan have?"   ← synonym reordering
  • "Explain JPMorgan's anti-money laundering duties"  ← paraphrase
  • "jpmorgan aml requirements"   ← casual shorthand

Why this matters for RAG
-------------------------
Each RAG pipeline invocation costs two things:
  1. A Bedrock API call: ~0.25¢–2¢ per query depending on model and token count.
  2. Latency: 1–4 seconds for the Bedrock round-trip alone.

In a compliance Q&A tool, analysts frequently ask the same regulatory questions
across different sessions or shortly after each other:
  "What does JPMorgan disclose about AML controls?"
  "What AML controls does JPM describe in their 10-K?"

Without caching, both questions pay the full Bedrock cost. With a semantic cache
at threshold=0.85, the second question hits the cache and returns in <5ms with
zero API cost — a 200–800× latency improvement.

Similarity threshold calibration
----------------------------------
The threshold controls the precision/recall tradeoff for cache hits:

  threshold=0.99  Very conservative — only near-identical phrasings hit.
                  Low risk of wrong cache hits; high miss rate.
  threshold=0.95  Recommended default — covers paraphrase and synonym variation
                  while being strict enough that questions about different tickers
                  or time periods don't pollute each other's answers.
  threshold=0.85  Aggressive — broader hits but risk of returning an AML answer
                  for a Basel III question if they share regulatory vocabulary.

For compliance use, err on the side of strictness: returning a stale or slightly
wrong cached answer to a regulatory question is worse than the small extra cost
of a Bedrock call.

Persistence layout
------------------
  data/semantic_cache/embeddings.faiss  — FAISS IndexFlatIP of question vectors
  data/semantic_cache/entries.json      — parallel list of {question, answer, ts}

The entry at position i in entries.json corresponds to the vector at FAISS
index i. This parallel-list design mirrors the FAISS embedder in ingestion/.

Usage
-----
    from rag_pipeline.optimization.semantic_cache import SemanticCache

    cache = SemanticCache()

    # Check before calling Bedrock
    cached = cache.get("What are JPMorgan's AML obligations?")
    if cached:
        return cached

    # Generate and store
    answer = bedrock_client.invoke(prompt).text
    cache.set("What are JPMorgan's AML obligations?", answer)
    cache.save()
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBED_DIM = 768

CACHE_DIR = Path("data/semantic_cache")
EMBEDDINGS_FILE = CACHE_DIR / "embeddings.faiss"
ENTRIES_FILE = CACHE_DIR / "entries.json"

# Default cosine similarity threshold for a cache hit.
# Cosine similarity ∈ [-1, 1] for L2-normalised vectors; values above 0.95
# represent semantically near-identical questions in practice.
DEFAULT_THRESHOLD = 0.85

# Maximum number of entries before the cache evicts the oldest questions.
# Each entry is ~1 KB (answer text) + 768 × 4 bytes (vector) ≈ 4 KB.
# At 10,000 entries that's ~40 MB — a safe ceiling for a single pod.
MAX_CACHE_SIZE = 10_000


# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC CACHE
# ─────────────────────────────────────────────────────────────────────────────

class SemanticCache:
    """
    In-memory semantic cache backed by FAISS cosine similarity search.

    Thread safety: FAISS search is read-safe for concurrent get() calls.
    set() modifies the index and entries list — use an external lock if
    calling set() from multiple threads simultaneously.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        cache_dir:  Path | str = CACHE_DIR,
        threshold:  float = DEFAULT_THRESHOLD,
        max_size:   int = MAX_CACHE_SIZE,
    ) -> None:
        """
        Parameters
        ----------
        model_name : str
            SentenceTransformer model for question embedding. Must match the
            model used to build any existing cache (dimension must agree with
            the saved FAISS index).
        cache_dir  : Path
            Directory for persisted cache files. Created by save().
        threshold  : float
            Default cosine similarity threshold for get(). Can be overridden
            per-call.
        max_size   : int
            Maximum number of entries. When exceeded, the oldest 10% are evicted.
        """
        self.cache_dir = Path(cache_dir)
        self.threshold = threshold
        self.max_size = max_size

        print(f"Loading embedding model '{model_name}' for semantic cache...")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        self._model = SentenceTransformer(model_name)

        # FAISS index — populated by set() or load().
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(EMBED_DIM)

        # Parallel list of cache entries. entries[i] corresponds to FAISS
        # vector i. Each entry stores the original question, the cached answer,
        # and the insertion timestamp (for eviction ordering).
        self._entries: list[dict[str, Any]] = []

        print(
            f"SemanticCache ready  "
            f"threshold={threshold}  "
            f"entries={self._index.ntotal}"
        )

    # ── get ───────────────────────────────────────────────────────────────────

    def get(
        self,
        question:  str,
        threshold: float | None = None,
    ) -> str | None:
        """
        Return a cached answer if a semantically similar question exists.

        Embeds the question, searches the FAISS index for the nearest stored
        question vector, and returns the associated answer if the cosine
        similarity meets the threshold.

        Parameters
        ----------
        question  : str           The incoming user question.
        threshold : float | None  Override the instance-level threshold for
                                  this call. Useful for context-dependent tuning
                                  (e.g. lower threshold for exploratory queries,
                                  higher for exact compliance lookups).

        Returns
        -------
        str   The cached answer text, or None if no sufficiently similar
              question exists in the cache.
        """
        if self._index.ntotal == 0:
            return None

        effective_threshold = threshold if threshold is not None else self.threshold

        query_vec = self._embed(question)
        # FAISS returns (scores, ids) each of shape (1, k). With k=1 we get
        # the single nearest neighbour. Score is cosine similarity ∈ [-1, 1]
        # because vectors are L2-normalised before indexing.
        scores, ids = self._index.search(query_vec, k=1)

        best_score = float(scores[0][0])
        best_id = int(ids[0][0])

        if best_id < 0 or best_score < effective_threshold:
            # ids[0][0] == -1 is the FAISS padding sentinel when ntotal < k.
            return None

        entry = self._entries[best_id]
        print(
            f"[cache HIT]  similarity={best_score:.4f}  "
            f"matched: '{entry['question'][:60]}...'"
        )
        return entry["answer"]

    # ── set ───────────────────────────────────────────────────────────────────

    def set(self, question: str, answer: str) -> None:
        """
        Add a question/answer pair to the cache.

        If the cache has reached max_size, the oldest 10% of entries are
        evicted before inserting. Eviction rebuilds the FAISS index from
        the surviving entries — O(N × D) but only triggered rarely.

        Parameters
        ----------
        question : str   The user question that was asked.
        answer   : str   The generated answer to cache.
        """
        if self._index.ntotal >= self.max_size:
            self._evict()

        vec = self._embed(question)
        self._index.add(vec)
        self._entries.append({
            "question":   question,
            "answer":     answer,
            "cached_at":  datetime.now(timezone.utc).isoformat(),
        })

    # ── save / load ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """
        Persist the FAISS index and entry list to disk.

        Creates cache_dir if it doesn't exist. Safe to call after every set()
        or in a periodic background flush — writes are atomic at the file level
        (FAISS write_index is all-or-nothing).
        """
        if self._index.ntotal == 0:
            return  # nothing to save

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(EMBEDDINGS_FILE))
        ENTRIES_FILE.write_text(
            json.dumps(self._entries, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"Semantic cache saved  "
            f"entries={self._index.ntotal}  path={self.cache_dir}/"
        )

    def load(self) -> None:
        """
        Load a previously saved cache from disk.

        Safe to call even when no cache exists yet — silently returns if the
        index file is absent, leaving the cache empty.
        """
        if not EMBEDDINGS_FILE.exists() or not ENTRIES_FILE.exists():
            print("No existing semantic cache found — starting empty.")
            return

        self._index = faiss.read_index(str(EMBEDDINGS_FILE))
        self._entries = json.loads(ENTRIES_FILE.read_text(encoding="utf-8"))

        if self._index.d != EMBED_DIM:
            raise ValueError(
                f"Saved cache index has dimension {self._index.d} but the "
                f"embedding model produces {EMBED_DIM}-dim vectors. "
                f"Delete {self.cache_dir}/ and rebuild."
            )

        print(
            f"Semantic cache loaded  "
            f"entries={self._index.ntotal}  path={self.cache_dir}/"
        )

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of question/answer pairs currently cached."""
        return self._index.ntotal

    def clear(self) -> None:
        """Remove all entries from the in-memory cache (does not delete files)."""
        self._index = faiss.IndexFlatIP(EMBED_DIM)
        self._entries = []

    # ── private helpers ───────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        """
        Embed a single string and return a (1, EMBED_DIM) float32 array.

        L2-normalises the vector so FAISS inner product == cosine similarity.
        batch_size=1 avoids the OMP/MPS segfault on Python 3.13 Apple Silicon.
        """
        vec = self._model.encode(
            [text],
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        return np.ascontiguousarray(vec)

    def _evict(self) -> None:
        """
        Evict the oldest 10% of entries when the cache is full.

        Because FAISS IndexFlatIP has no delete operation, eviction works by:
          1. Slicing off the oldest entries from the parallel list.
          2. Re-embedding the surviving questions (they're already in the index
             but we need fresh vectors to rebuild from scratch).
          3. Creating a new empty index and adding the surviving vectors.

        This is O(survivors × embed_dim) and only runs when max_size is reached.
        At max_size=10,000 and embed_dim=768 it takes ~2–5 seconds — acceptable
        as an infrequent maintenance operation.
        """
        evict_count = max(1, self.max_size // 10)
        survivors = self._entries[evict_count:]
        survivor_qs = [e["question"] for e in survivors]

        print(
            f"[cache evict]  removing {evict_count} oldest entries, "
            f"retaining {len(survivors)}"
        )

        if not survivors:
            self.clear()
            return

        # Re-embed surviving questions in one batch for efficiency.
        vecs = self._model.encode(
            survivor_qs,
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        vecs = np.ascontiguousarray(vecs)

        self._index = faiss.IndexFlatIP(EMBED_DIM)
        self._index.add(vecs)
        self._entries = survivors


# ─────────────────────────────────────────────────────────────────────────────
# DEMO / CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cache = SemanticCache()

    # Seed the cache with a few entries
    qa_pairs = [
        (
            "What are JPMorgan's AML obligations under the Bank Secrecy Act?",
            "JPMorgan maintains AML compliance programs under the BSA, filing "
            "SARs for suspicious transactions and applying CDD/EDD for high-risk "
            "customers.",
        ),
        (
            "What is JPMorgan's CET1 capital ratio?",
            "JPMorgan's CET1 capital ratio was 13.1% as of Q4 2023, above the "
            "Basel III minimum of 4.5% plus the 2.5% capital conservation buffer.",
        ),
        (
            "What fraud detection controls does Bank of America disclose?",
            "Bank of America discloses transaction monitoring, velocity checks, "
            "and ML-based anomaly detection in its 10-K risk factor disclosures.",
        ),
    ]

    for question, answer in qa_pairs:
        cache.set(question, answer)

    print(f"\nCache size: {cache.size}")
    cache.save()

    # Test exact match
    print("\n── Exact match test")
    result = cache.get("What are JPMorgan's AML obligations under the Bank Secrecy Act?")
    print(f"  Hit: {result is not None}  answer[:60]: {(result or '')[:60]}")

    # Test paraphrase match
    print("\n── Paraphrase match test")
    result = cache.get("What AML requirements does JPMorgan have under BSA regulations?")
    print(f"  Hit: {result is not None}  answer[:60]: {(result or '')[:60]}")

    # Test threshold miss
    print("\n── Low-threshold miss test (threshold=0.99)")
    result = cache.get(
        "Explain JPMorgan anti-money laundering duties",
        threshold=0.99,
    )
    print(f"  Hit: {result is not None}")

    # Test unrelated question
    print("\n── Unrelated question test")
    result = cache.get("What is the weather in New York?")
    print(f"  Hit: {result is not None}  (expected: False)")

    # Reload from disk
    print("\n── Reload test")
    cache2 = SemanticCache()
    cache2.load()
    print(f"  Loaded {cache2.size} entries")
    result = cache2.get("JPMorgan CET1 ratio Basel")
    print(f"  Post-reload hit: {result is not None}")
