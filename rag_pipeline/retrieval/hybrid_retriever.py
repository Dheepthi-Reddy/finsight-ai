"""
Hybrid retrieval for the FinSight RAG pipeline.

Why hybrid beats dense-only retrieval
--------------------------------------
Dense retrieval (FAISS + sentence embeddings) excels at *semantic* matching:
it finds passages whose *meaning* is similar to the query even when they share
no words. Ask "BSA compliance obligations" and it correctly retrieves a chunk
about "anti-money laundering requirements" because both phrases map to nearby
vectors in the 768-dimensional embedding space.

But dense retrieval has a blind spot: *exact-match* queries. Regulatory text
is full of precise identifiers — "CET1", "SAR", "12 CFR 326.8", ticker symbols,
specific dollar figures — where the exact string matters and paraphrases don't
exist. A query for "OCC bulletin 2023-17" should retrieve the one chunk that
mentions "OCC bulletin 2023-17", not the semantically closest passage about
"regulatory guidance". Dense models often blur these distinctions because rare
tokens receive weak representation during pre-training.

Sparse retrieval (BM25) excels at exactly this: it scores documents by *term
frequency × inverse document frequency*, so a chunk that contains the exact
query tokens — however rare — scores very high. BM25 has no notion of meaning,
only of word overlap, but that is precisely the property we want for exact-match
regulatory lookups.

Combining both:
  • Dense covers semantic similarity — handles paraphrase, synonyms, concept
    matching (finds the right section even when phrased differently).
  • Sparse covers exact-match precision — pins down specific codes, names,
    figures, and verbatim terminology.
  • Together they address each other's failure modes without hyperparameter
    tuning on the blend ratio, thanks to Reciprocal Rank Fusion.

What Reciprocal Rank Fusion (RRF) does
----------------------------------------
RRF is a rank-aggregation algorithm that merges two or more ranked lists into
a single combined ranking without needing to know or normalise the score scales
of the individual retrievers.

For each document d, its RRF score is computed as:

    rrf_score(d) = Σ  1 / (k + rank_i(d))
                  i

where:
  • rank_i(d) is the 1-based position of document d in retriever i's results
    (documents absent from a list are treated as rank = ∞, contributing 0).
  • k is a smoothing constant (typically 60) that dampens the advantage of the
    top rank: rank 1 contributes 1/(60+1) ≈ 0.0164, rank 10 contributes
    1/(60+10) ≈ 0.0143. The difference is modest, avoiding winner-take-all
    behaviour while still honouring rank order.

Why RRF over score normalisation
----------------------------------
An alternative is to normalise each retriever's raw scores to [0, 1] and take
a weighted sum. The problem: FAISS cosine similarities and BM25 TF-IDF scores
live in incompatible spaces. Cosine similarity is bounded [−1, 1] for
normalised vectors; BM25 scores are unbounded and depend on corpus statistics.
Any fixed normalisation scheme would need careful per-corpus calibration.

RRF sidesteps this entirely by operating only on *ranks*, which are naturally
comparable across any retriever. It was introduced by Cormack, Clarke & Buettcher
(SIGIR 2009) and has been empirically shown to outperform score fusion on most
IR benchmarks, especially when the retrievers have different score distributions.

Persistence layout
------------------
  data/hybrid_index/faiss.index      — FAISS binary index (dense vectors)
  data/hybrid_index/metadata.json    — parallel chunk list (text + provenance)
  data/hybrid_index/bm25_corpus.json — tokenised BM25 corpus (list of token lists)

The BM25 index itself is not serialised (rank_bm25 provides no built-in save);
instead the tokenised corpus is persisted and the index is reconstructed from it
on load — cheap because BM25 fitting is O(N × avg_doc_len).

Usage
-----
    from rag_pipeline.retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever()
    retriever.build_index(chunks)          # chunks from chunker.py
    results = retriever.search("AML SAR filings JPMorgan", top_k=5)

    # Persist and reload
    retriever.save()
    retriever2 = HybridRetriever()
    retriever2.load()
    results = retriever2.search("Basel III CET1 ratio", top_k=5)
"""

from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False
    import warnings
    warnings.warn(
        "rank_bm25 not installed — BM25 sparse retrieval will be unavailable. "
        "Install with: pip install rank-bm25",
        ImportWarning,
        stacklevel=1,
    )

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBED_DIM = 768

INDEX_DIR = Path("data/hybrid_index")
FAISS_FILE = INDEX_DIR / "faiss.index"
METADATA_FILE = INDEX_DIR / "metadata.json"
BM25_CORPUS_FILE = INDEX_DIR / "bm25_corpus.json"

# RRF smoothing constant. k=60 is the value recommended in the original paper
# and is the de-facto default in production hybrid retrieval systems (e.g.
# Elasticsearch, Weaviate, LangChain). Increasing k flattens the rank curve
# (top ranks matter less); decreasing k sharpens it.
RRF_K = 60

# Batch size for sentence-transformer encoding during index build.
# Kept at 1 for Python 3.13 / Apple Silicon segfault avoidance (see embedder.py).
ENCODE_BATCH_SIZE = 1

# Stopwords removed during BM25 tokenisation. A minimal set focused on
# function words that carry no retrieval signal in financial/legal text.
# We keep short tokens that ARE meaningful in this domain (e.g. "act", "sec")
# by using a domain-aware list rather than a generic NLP stopword list.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "that", "this", "these", "those",
    "it", "its", "as", "not", "no", "nor", "so", "yet", "both", "either",
    "each", "more", "most", "other", "such", "than", "too", "very", "just",
})


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID RETRIEVER
# ─────────────────────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Combines FAISS dense retrieval and BM25 sparse retrieval, fused with RRF.

    Instantiate once at application startup. build_index() populates both the
    FAISS index and the BM25 index from a list of chunk dicts. After that,
    search() runs both retrievers in parallel (Python in-process) and returns
    RRF-fused results.

    Thread safety: FAISS search and BM25 scoring are both read-only after
    indexing. The instance holds no mutable per-request state, so concurrent
    search() calls are safe.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        index_dir: Path | str = INDEX_DIR,
        rrf_k: int = RRF_K,
    ) -> None:
        """
        Parameters
        ----------
        model_name : str
            SentenceTransformer model for dense encoding. Must match the model
            used to build an existing index if load() will be called.
        index_dir  : Path | str
            Directory for persisted index files. Created automatically by save().
        rrf_k      : int
            RRF smoothing constant. Use 60 unless you have a strong empirical
            reason to deviate.
        """
        self.index_dir = Path(index_dir)
        self.rrf_k = rrf_k

        print(f"Loading embedding model '{model_name}'...")
        import os
        # Prevent segfault on Python 3.13 + Apple Silicon (see embedder.py).
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        self._model = SentenceTransformer(model_name)

        # Dense index and metadata — populated by build_index() or load().
        self._faiss_index: faiss.IndexFlatIP | None = None
        self._metadata: list[dict[str, Any]] = []

        # Sparse index — populated by build_index() or load().
        # _bm25_corpus is the tokenised representation persisted to disk;
        # _bm25 is the BM25Okapi object reconstructed from it at load time.
        self._bm25: BM25Okapi | None = None
        self._bm25_corpus: list[list[str]] = []

        print("HybridRetriever ready (index not yet built — call build_index() or load())")

    # ── Build ─────────────────────────────────────────────────────────────────

    def build_index(self, chunks: list[dict[str, Any]]) -> None:
        """
        Index a list of document chunks in both FAISS and BM25.

        Both indexes are built from scratch on each call — existing index data
        is replaced. This is appropriate for a full rebuild after ingestion.
        For incremental updates, call build_index() with the full combined
        corpus rather than just new chunks (FAISS IndexFlatIP has no delete
        operation, and BM25 requires the full corpus to compute IDF correctly).

        Parameters
        ----------
        chunks : list[dict]
            Each dict must have keys: text, metadata.
            Produced by rag_pipeline.ingestion.chunker.chunk_documents().
        """
        if not chunks:
            raise ValueError("chunks list is empty — nothing to index")

        texts = [c["text"] for c in chunks]
        print(f"Building hybrid index for {len(texts):,} chunks...")

        # ── Dense: FAISS ─────────────────────────────────────────────────────
        # Encode all texts and L2-normalise so inner product == cosine similarity.
        print("  Encoding with SentenceTransformer (dense)...")
        vectors = self._model.encode(
            texts,
            batch_size=ENCODE_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        vectors = np.ascontiguousarray(vectors)

        self._faiss_index = faiss.IndexFlatIP(EMBED_DIM)
        self._faiss_index.add(vectors)

        # ── Sparse: BM25 ─────────────────────────────────────────────────────
        # Tokenise each chunk into a list of lowercase content tokens. BM25Okapi
        # expects a list-of-lists; each inner list is the tokenised document.
        print("  Tokenising for BM25 (sparse)...")
        self._bm25_corpus = [_tokenise(t) for t in texts]
        if _BM25_AVAILABLE:
            self._bm25 = BM25Okapi(self._bm25_corpus)
        else:
            self._bm25 = None
            print("  WARNING: rank_bm25 not installed — BM25 index skipped")

        # ── Metadata ─────────────────────────────────────────────────────────
        # Parallel list: metadata[i] ↔ FAISS vector i ↔ bm25_corpus[i].
        self._metadata = []
        for chunk in chunks:
            self._metadata.append({"text": chunk["text"], **chunk["metadata"]})

        print(
            f"Hybrid index built  "
            f"dense={self._faiss_index.ntotal:,} vectors  "
            f"sparse={'enabled' if self._bm25 else 'disabled (rank_bm25 missing)'}"
        )

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Retrieve the top_k most relevant chunks using RRF-fused hybrid search.

        Steps
        -----
        1. Run dense FAISS search with the encoded query → ranked list A.
        2. Run BM25 sparse search with the tokenised query → ranked list B.
        3. Apply Reciprocal Rank Fusion to merge A and B into a single ranking.
        4. Return the top_k results, each annotated with its RRF score and
           the individual dense/sparse ranks for transparency.

        If BM25 is unavailable (rank_bm25 not installed), falls back to dense-
        only retrieval and logs a warning.

        Parameters
        ----------
        query  : str   Natural-language retrieval query.
        top_k  : int   Number of results to return.

        Returns
        -------
        list[dict]
            Keys: text, rrf_score, dense_rank, sparse_rank,
                  ticker, filing_type, source, chunk_index, chunk_count.
            Sorted by rrf_score descending (highest relevance first).
        """
        if self._faiss_index is None or not self._metadata:
            raise RuntimeError(
                "Index not built. Call build_index(chunks) or load() first."
            )

        n_docs = len(self._metadata)
        fetch_k = min(top_k * 3, n_docs)  # over-fetch for post-fusion trimming

        # ── Step 1: Dense retrieval ───────────────────────────────────────────
        # Encode and normalise the query with the same settings used at index
        # time so the query vector lives in the same cosine-similarity space.
        query_vec = self._model.encode(
            [query],
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        # FAISS returns (scores, ids) each of shape (1, fetch_k).
        # Scores are cosine similarities ∈ [-1, 1] for normalised vectors.
        faiss_scores, faiss_ids = self._faiss_index.search(query_vec, fetch_k)

        # Build a dict: doc_index → 1-based rank in the dense results.
        # FAISS already returns results sorted best-first, so rank = position + 1.
        dense_ranks: dict[int, int] = {}
        for rank, idx in enumerate(faiss_ids[0], start=1):
            if idx >= 0:  # FAISS uses -1 as padding sentinel
                dense_ranks[int(idx)] = rank

        # ── Step 2: Sparse retrieval (BM25) ──────────────────────────────────
        # BM25Okapi.get_scores() returns a float array of length N (corpus size),
        # one score per document. We argsort descending and take the top fetch_k.
        sparse_ranks: dict[int, int] = {}
        if self._bm25 is not None:
            query_tokens = _tokenise(query)
            if query_tokens:
                bm25_scores = self._bm25.get_scores(query_tokens)
                # argsort ascending, then reverse for descending order.
                top_bm25_ids = np.argsort(bm25_scores)[::-1][:fetch_k]
                for rank, idx in enumerate(top_bm25_ids, start=1):
                    if bm25_scores[idx] > 0:  # only include docs with any match
                        sparse_ranks[int(idx)] = rank
        else:
            # Graceful degradation: hybrid search without BM25 is just dense search.
            import warnings
            warnings.warn(
                "BM25 index unavailable — returning dense-only results. "
                "Install rank-bm25 for full hybrid retrieval.",
                RuntimeWarning,
                stacklevel=2,
            )

        # ── Step 3: Reciprocal Rank Fusion ────────────────────────────────────
        # Collect all candidate doc indices from either retriever.
        all_candidates = set(dense_ranks.keys()) | set(sparse_ranks.keys())

        rrf_scores: dict[int, float] = {}
        for doc_idx in all_candidates:
            score = 0.0
            # Dense contribution: 0 if document wasn't in dense results.
            if doc_idx in dense_ranks:
                score += 1.0 / (self.rrf_k + dense_ranks[doc_idx])
            # Sparse contribution: 0 if document wasn't in sparse results.
            if doc_idx in sparse_ranks:
                score += 1.0 / (self.rrf_k + sparse_ranks[doc_idx])
            rrf_scores[doc_idx] = score

        # Sort by RRF score descending and take top_k.
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        ranked = ranked[:top_k]

        # ── Step 4: Assemble results ──────────────────────────────────────────
        results = []
        for doc_idx, rrf_score in ranked:
            meta = self._metadata[doc_idx]
            results.append({
                "text": meta["text"],
                "rrf_score": round(rrf_score, 6),
                # Individual ranks included for transparency / debugging.
                # None means the document didn't appear in that retriever's list.
                "dense_rank": dense_ranks.get(doc_idx),
                "sparse_rank": sparse_ranks.get(doc_idx),
                "ticker": meta.get("ticker", ""),
                "filing_type": meta.get("filing_type", ""),
                "source": meta.get("source", ""),
                "chunk_index": meta.get("chunk_index", -1),
                "chunk_count": meta.get("chunk_count", -1),
            })

        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """
        Persist the hybrid index to disk.

        Writes three files to index_dir/:
          faiss.index      — binary FAISS index (dense vectors, not human-readable)
          metadata.json    — chunk text + provenance for all indexed documents
          bm25_corpus.json — tokenised corpus; BM25Okapi is reconstructed from
                             this at load time (rank_bm25 has no native serialisation)

        Raises RuntimeError if build_index() hasn't been called yet.
        """
        if self._faiss_index is None:
            raise RuntimeError("Nothing to save — call build_index() first.")

        self.index_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._faiss_index, str(FAISS_FILE))
        METADATA_FILE.write_text(
            json.dumps(self._metadata, ensure_ascii=False), encoding="utf-8"
        )
        BM25_CORPUS_FILE.write_text(
            json.dumps(self._bm25_corpus, ensure_ascii=False), encoding="utf-8"
        )

        print(
            f"Hybrid index saved to {self.index_dir}/  "
            f"({self._faiss_index.ntotal:,} vectors, "
            f"{len(self._bm25_corpus):,} BM25 docs)"
        )

    def load(self) -> None:
        """
        Load a previously saved hybrid index from disk.

        Reconstructs the FAISS index, metadata, and BM25Okapi index from the
        three persisted files. Call this at application startup instead of
        build_index() when the corpus hasn't changed since the last save().

        Raises FileNotFoundError if any of the three index files are missing.
        Raises ValueError if the saved FAISS dimension doesn't match EMBED_DIM
        (which would indicate a model mismatch — silent mismatches produce
        garbage retrieval results).
        """
        for path in (FAISS_FILE, METADATA_FILE, BM25_CORPUS_FILE):
            if not path.exists():
                raise FileNotFoundError(
                    f"Index file not found: {path}\n"
                    "Run build_index(chunks) followed by save() to create it."
                )

        print(f"Loading hybrid index from {self.index_dir}/...")

        # Dense index
        self._faiss_index = faiss.read_index(str(FAISS_FILE))
        if self._faiss_index.d != EMBED_DIM:
            raise ValueError(
                f"Saved FAISS index has dimension {self._faiss_index.d}, but "
                f"'{MODEL_NAME}' produces {EMBED_DIM}-dim vectors. "
                f"Delete {self.index_dir}/ and rebuild."
            )

        # Metadata
        self._metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))

        # BM25: reload tokenised corpus and reconstruct the index.
        self._bm25_corpus = json.loads(BM25_CORPUS_FILE.read_text(encoding="utf-8"))
        if _BM25_AVAILABLE:
            self._bm25 = BM25Okapi(self._bm25_corpus)
        else:
            self._bm25 = None
            import warnings
            warnings.warn(
                "rank_bm25 not installed — BM25 component unavailable after load. "
                "Install with: pip install rank-bm25",
                ImportWarning,
                stacklevel=2,
            )

        print(
            f"Hybrid index loaded  "
            f"dense={self._faiss_index.ntotal:,} vectors  "
            f"sparse={'enabled' if self._bm25 else 'disabled'}"
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def doc_count(self) -> int:
        """Number of documents currently indexed."""
        return len(self._metadata)

    @property
    def is_ready(self) -> bool:
        """True when both the FAISS index and metadata are populated."""
        return self._faiss_index is not None and bool(self._metadata)


# ─────────────────────────────────────────────────────────────────────────────
# TOKENISATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> list[str]:
    """
    Tokenise text into lowercase content tokens for BM25 indexing/querying.

    Strategy
    --------
    1. Lowercase the entire string.
    2. Split on whitespace and punctuation (keeping hyphenated compounds like
       "anti-money-laundering" intact — hyphens are meaningful in financial text).
    3. Strip leading/trailing punctuation from each token.
    4. Drop stopwords and tokens shorter than 2 characters.

    We deliberately preserve:
    - Numeric tokens ("12", "4.5%", "2023") — figures are important in regulatory
      text ("CET1 ratio of 4.5%").
    - Acronyms ("AML", "BSA", "SAR", "OCC") — these are precise identifiers that
      BM25 should match exactly.
    - Hyphenated terms split on hyphens so "anti-money-laundering" contributes
      three tokens: "anti", "money", "laundering".

    Parameters
    ----------
    text : str   Raw document or query text.

    Returns
    -------
    list[str]   Filtered token list ready for BM25Okapi.
    """
    text = text.lower()

    # Replace hyphens with spaces so hyphenated compounds are split into
    # individual tokens (each component carries BM25 signal independently).
    text = text.replace("-", " ")

    # Split on any non-alphanumeric character (punctuation, whitespace, etc.).
    # re.split with a character class is faster than str.split for mixed delimiters.
    raw_tokens = re.split(r"[^a-z0-9.%]+", text)

    tokens = []
    for token in raw_tokens:
        # Strip any residual punctuation attached to numeric tokens ("12%", "4.5.").
        token = token.strip(string.punctuation)
        if len(token) < 2:
            continue
        if token in _STOPWORDS:
            continue
        tokens.append(token)

    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# DEMO / CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rag_pipeline.ingestion.chunker import chunk_documents
    from rag_pipeline.ingestion.sec_loader import load_filings

    print("Loading stub filings...")
    docs = load_filings(force_stubs=True)
    chunks = chunk_documents(docs)

    retriever = HybridRetriever()
    retriever.build_index(chunks)
    retriever.save()

    queries = [
        # Semantic query — dense retrieval should dominate
        "What AML obligations do banks have under anti-money laundering regulations?",
        # Exact-match query — sparse retrieval should contribute
        "SAR suspicious activity report filing requirements Bank Secrecy Act",
        # Mixed — both retrievers contribute
        "JPMorgan CET1 capital ratio Basel III 2023",
    ]

    for query in queries:
        print(f"\n{'─' * 68}")
        print(f"Query: {query}")
        results = retriever.search(query, top_k=3)
        for r in results:
            dense_r = f"#{r['dense_rank']}" if r["dense_rank"] else " —"
            sparse_r = f"#{r['sparse_rank']}" if r["sparse_rank"] else " —"
            print(
                f"  rrf={r['rrf_score']:.5f}  "
                f"dense={dense_r:>3}  sparse={sparse_r:>3}  "
                f"{r['ticker']:>4}/{r['filing_type']:<4}  "
                f"{r['text'][:80].strip()}..."
            )

    # ── Reload demo ───────────────────────────────────────────────────────────
    print(f"\n{'─' * 68}")
    print("Reloading index from disk...")
    retriever2 = HybridRetriever()
    retriever2.load()
    print(f"Loaded {retriever2.doc_count:,} documents")
    results = retriever2.search("Basel III capital requirements", top_k=2)
    for r in results:
        print(f"  rrf={r['rrf_score']:.5f}  {r['ticker']}/{r['filing_type']}  "
              f"{r['text'][:80].strip()}...")
