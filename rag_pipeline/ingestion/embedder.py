"""
Embeds document chunks into a FAISS vector index for semantic retrieval.

What embeddings are
-------------------
An embedding is a fixed-length numeric vector (here, 768 floats) that
encodes the *meaning* of a piece of text as a point in high-dimensional space.
Texts with similar meaning are mapped to nearby points regardless of exact
wording — so "anti-money laundering requirements" and "BSA compliance
obligations" produce vectors that point in nearly the same direction even
though they share no words. This is what makes semantic search possible.

How FAISS search works
----------------------
FAISS (Facebook AI Similarity Search) stores all document vectors in memory.
Given a query vector, it finds the k stored vectors with the highest inner
product (dot product). With L2-normalised vectors, inner product equals cosine
similarity: 1.0 = identical direction, 0.0 = orthogonal (unrelated).

IndexFlatIP does exhaustive exact search — every stored vector is scored
against the query in a single vectorised C++ kernel. At our scale (~50k
chunks × 768 dims) this takes <5ms on CPU. If the corpus grows to millions
of vectors, swap to IndexIVFFlat or IndexHNSWFlat for approximate search.

Persistence layout
------------------
  data/faiss_index/index.faiss    — binary FAISS index (vectors only)
  data/faiss_index/metadata.json  — parallel list of chunk text + metadata
  (integer position in metadata list == FAISS vector ID)

Usage
-----
    from rag_pipeline.ingestion.embedder import FAISSEmbedder

    embedder = FAISSEmbedder()                  # load once at startup
    embedder.upsert(chunks)                     # list from chunker.py
    results = embedder.query("AML obligations for JPM", top_k=5)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Configuration ─────────────────────────────────────────────────────────────
# all-mpnet-base-v2 scores highest on semantic textual similarity benchmarks
# among freely available models and produces 768-dim vectors that capture
# nuanced financial/legal language better than the smaller MiniLM variants.
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBED_DIM  = 768   # fixed by the model architecture — must match stored index

INDEX_DIR     = Path("data/faiss_index")
INDEX_FILE    = INDEX_DIR / "index.faiss"
METADATA_FILE = INDEX_DIR / "metadata.json"

# 64 chunks × ~512 chars ≈ modest GPU/CPU batch; larger batches improve
# throughput but increase peak RAM proportionally.
ENCODE_BATCH_SIZE = 64


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDER
# ─────────────────────────────────────────────────────────────────────────────

class FAISSEmbedder:
    """
    Single-instance wrapper around a SentenceTransformer encoder and a FAISS
    inner-product index. Instantiate once at application startup (model load
    takes ~2s on first call; the local cache makes subsequent loads instant).
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        index_dir:  Path | str = INDEX_DIR,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading embedding model '{model_name}'...")
        # SentenceTransformer downloads the model weights on first use and
        # caches them in ~/.cache/torch/sentence_transformers/
        self.model = SentenceTransformer(model_name)

        # metadata[i] holds the text and provenance for the vector at FAISS
        # position i. FAISS only stores float arrays — all text lives here.
        self._metadata: list[dict[str, Any]] = []
        self._index = self._load_or_create()

        print(
            f"FAISSEmbedder ready  "
            f"model={model_name}  "
            f"vectors={self._index.ntotal:,}  "
            f"dim={EMBED_DIM}"
        )

    # ── upsert ────────────────────────────────────────────────────────────────

    def upsert(self, chunks: list[dict[str, Any]]) -> None:
        """
        Encode chunks and append their vectors to the FAISS index.

        IndexFlatIP has no native delete or update operation — vectors are
        stored as a flat array and the only mutating operation is append.
        Avoid re-ingesting the same filing twice; if a full rebuild is needed
        (e.g. after model upgrade), delete data/faiss_index/ and re-run.

        Parameters
        ----------
        chunks : list[dict]
            Each dict must have keys: text, metadata.
            Produced by rag_pipeline.ingestion.chunker.chunk_documents().
        """
        if not chunks:
            return

        texts = [c["text"] for c in chunks]

        # Encode all texts and L2-normalise so inner product == cosine similarity.
        # normalize_embeddings=True applies the normalisation inside the model's
        # encode() call — faster than a separate faiss.normalize_L2() pass.
        print(f"Embedding {len(texts):,} chunks (batch_size={ENCODE_BATCH_SIZE})...")
        vectors = self.model.encode(
            texts,
            batch_size=ENCODE_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        # FAISS requires a C-contiguous float32 array; .astype() may already
        # return one, but np.ascontiguousarray guarantees it.
        vectors = np.ascontiguousarray(vectors)

        self._index.add(vectors)

        # Append text + metadata at the same positions as the new vectors.
        # Keeping text here avoids a second disk lookup when returning results.
        for chunk in chunks:
            self._metadata.append({"text": chunk["text"], **chunk["metadata"]})

        self._save()
        print(
            f"Upsert complete: {len(chunks):,} added  "
            f"total={self._index.ntotal:,} vectors"
        )

    # ── query ─────────────────────────────────────────────────────────────────

    def query(self, text: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Return the top_k chunks most semantically similar to the query text.

        The query is embedded and normalised identically to stored chunks, then
        FAISS computes the inner product against every stored vector in a single
        C++ BLAS call. Results arrive pre-sorted highest-score-first.

        Score interpretation (cosine similarity after normalisation):
          ≥ 0.80 — very strong match, essentially same concept
          0.60–0.79 — strong match, highly relevant
          0.40–0.59 — moderate match, probably relevant
          < 0.40  — weak match, likely noise for this query

        Parameters
        ----------
        text : str
            Natural-language query, e.g. "JPMorgan AML suspicious activity".
        top_k : int
            Number of results to return. Capped at index size automatically.

        Returns
        -------
        list[dict]
            Keys: text, score, ticker, filing_type, source,
                  chunk_index, chunk_count.
            Sorted by score descending.
        """
        if self._index.ntotal == 0:
            return []

        top_k = min(top_k, self._index.ntotal)

        # Embed and normalise the query with the same settings as upsert()
        # so the vector lives in the same normalised space as stored chunks.
        query_vec = self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        # index.search returns (scores, ids) each of shape (1, top_k).
        # FAISS uses id=-1 as a padding sentinel when ntotal < top_k.
        scores, ids = self._index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            meta = self._metadata[idx]
            results.append({
                "text":        meta["text"],
                "score":       round(float(score), 4),
                "ticker":      meta.get("ticker", ""),
                "filing_type": meta.get("filing_type", ""),
                "source":      meta.get("source", ""),
                "chunk_index": meta.get("chunk_index", -1),
                "chunk_count": meta.get("chunk_count", -1),
            })

        return results  # already sorted by FAISS

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def vector_count(self) -> int:
        """Total number of vectors currently stored in the index."""
        return self._index.ntotal

    # ── persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        """
        Persist the FAISS index and metadata to disk.

        Called automatically after every upsert() so the store survives
        process restarts. For very high-frequency ingestion pipelines,
        consider batching saves (call _save() every N upserts instead).
        """
        faiss.write_index(self._index, str(INDEX_FILE))
        METADATA_FILE.write_text(json.dumps(self._metadata, ensure_ascii=False))

    def _load_or_create(self) -> faiss.IndexFlatIP:
        """Load an existing index from disk or create a new empty one."""
        if INDEX_FILE.exists() and METADATA_FILE.exists():
            print(f"Loading existing index from {self.index_dir}/")
            index = faiss.read_index(str(INDEX_FILE))

            # Dimension mismatch means the index was built with a different
            # model — a silent mismatch would produce garbage retrieval.
            if index.d != EMBED_DIM:
                raise ValueError(
                    f"Saved index has dimension {index.d}, but "
                    f"'{MODEL_NAME}' produces {EMBED_DIM}-dim vectors. "
                    f"Delete {self.index_dir}/ and rebuild the index."
                )

            self._metadata = json.loads(METADATA_FILE.read_text())
            return index

        # IndexFlatIP: exact cosine similarity search for L2-normalised vectors.
        # No training required — vectors can be added immediately after creation.
        print(f"Creating new FAISSIndexFlatIP (dim={EMBED_DIM})...")
        return faiss.IndexFlatIP(EMBED_DIM)


# ─────────────────────────────────────────────────────────────────────────────
# CLI — full ingestion pipeline demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rag_pipeline.ingestion.chunker import chunk_documents
    from rag_pipeline.ingestion.sec_loader import load_filings

    # Load stubs so the demo works without SEC network access
    docs   = load_filings(force_stubs=True)
    chunks = chunk_documents(docs)

    embedder = FAISSEmbedder()
    embedder.upsert(chunks)

    # ── Sample queries covering each major compliance topic in the stubs ──────
    queries = [
        "What are the AML and Bank Secrecy Act obligations for financial firms?",
        "How does JPMorgan manage suspicious activity report filings?",
        "What fraud detection controls do banks use for card-not-present transactions?",
        "Explain Basel III capital requirements and CET1 ratios.",
        "What happens when the OCC issues a consent order for AML deficiencies?",
    ]

    for q in queries:
        print(f"\n── Query: {q}")
        results = embedder.query(q, top_k=3)
        for r in results:
            print(
                f"  [{r['score']:.3f}] {r['ticker']} {r['filing_type']}  "
                f"chunk {r['chunk_index']}/{r['chunk_count']}"
            )
            print(f"  {r['text'][:120].strip()}...")
