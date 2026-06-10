"""
FinSight RAG pipeline — wires retrieval, generation, and evaluation together.

How RAG works end-to-end
--------------------------
RAG (Retrieval-Augmented Generation) is a pattern that gives a language model
access to a curated, up-to-date knowledge base without retraining it. The
pipeline has three stages that run in sequence for every user query:

  ┌─────────────────────────────────────────────────────────────────────┐
  │  1. RETRIEVAL                                                        │
  │                                                                      │
  │  User question                                                       │
  │       │                                                              │
  │       ▼                                                              │
  │  SentenceTransformer → 768-dim query vector                         │
  │       │                                                              │
  │       ▼                                                              │
  │  FAISS IndexFlatIP.search() → top-k chunks, cosine similarity       │
  │  (optional: filter by ticker to scope to one company)               │
  │       │                                                              │
  │  Why retrieval first: the full SEC corpus is millions of tokens —   │
  │  far more than any model's context window. Semantic search compresses│
  │  that to the 5–10 most relevant passages, typically <2,000 tokens.  │
  └──────────────────────────────┬──────────────────────────────────────┘
                                 │
  ┌──────────────────────────────▼──────────────────────────────────────┐
  │  2. AUGMENTATION                                                     │
  │                                                                      │
  │  Retrieved chunks → context string with source labels:             │
  │    [JPM / 10-K / Chunk 4 of 31]                                     │
  │    <chunk text>                                                      │
  │    [BAC / 10-Q / Chunk 7 of 28]                                     │
  │    <chunk text>                                                      │
  │       │                                                              │
  │       ▼                                                              │
  │  build_compliance_prompt(context, question)                         │
  │  → user message with embedded source passages                       │
  │                                                                      │
  │  The label format [TICKER / TYPE / Chunk N of M] mirrors the        │
  │  citation format required in COMPLIANCE_SYSTEM_PROMPT, so Claude    │
  │  can copy them verbatim into its answer without inventing new ones. │
  └──────────────────────────────┬──────────────────────────────────────┘
                                 │
  ┌──────────────────────────────▼──────────────────────────────────────┐
  │  3. GENERATION                                                       │
  │                                                                      │
  │  COMPLIANCE_SYSTEM_PROMPT (expert persona + citation rules)         │
  │  + user message (context + question)                                │
  │       │                                                              │
  │       ▼                                                              │
  │  BedrockClient.invoke() → Claude Sonnet (AWS Bedrock)               │
  │       │                                                              │
  │       ▼                                                              │
  │  response_evaluator.evaluate_response()                             │
  │  → BLEU / ROUGE-L / faithfulness scores                             │
  │                                                                      │
  │  Generation conditioned on the context means Claude cannot draw on  │
  │  training knowledge — the system prompt prohibits it. All claims    │
  │  must be traced to a chunk label, making hallucinations auditable.  │
  └─────────────────────────────────────────────────────────────────────┘

Why each component is loaded at class init rather than per-query
-----------------------------------------------------------------
  FAISSEmbedder : loads the SentenceTransformer model (~400 MB) and the
                  FAISS index from disk. Both are cheap to reuse across
                  thousands of queries. Re-loading per query would add 2–4s
                  of latency on every call.
  BedrockClient : creates a boto3 session and resolves AWS credentials.
                  Session creation is ~50ms; reusing it avoids that overhead
                  and shares the underlying connection pool.

Usage
-----
    from rag_pipeline.chain import FinSightChain

    chain = FinSightChain()
    result = chain.query("What are JPMorgan's AML obligations under the BSA?")
    print(result["answer"])
    print(result["scores"])

    # Scope to a single company
    result = chain.query("Describe capital ratio disclosures", ticker="JPM")
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any

from rag_pipeline.generation.bedrock_client import BedrockClient, InvokeResponse
from rag_pipeline.generation.prompt_templates import (
    COMPLIANCE_SYSTEM_PROMPT,
    build_compliance_prompt,
)
from rag_pipeline.generation.response_evaluator import EvaluationResult, evaluate_response
from rag_pipeline.ingestion.embedder import FAISSEmbedder

# ── Retrieval configuration ────────────────────────────────────────────────────
# Chunks below this cosine similarity are excluded even if they are in the
# top-k results. Cosine similarity ∈ [-1, 1] for normalised vectors; in
# practice SEC chunk similarities cluster between 0.20 and 0.85.
# 0.30 removes clearly unrelated passages while keeping borderline-relevant ones
# that may provide useful regulatory context.
MIN_SIMILARITY_SCORE = 0.30

# Hard cap on chunks sent to the model regardless of top_k argument.
# all-mpnet-base-v2 chunks average ~400 chars. 15 chunks × 400 chars ≈ 6,000
# chars ≈ ~1,500 tokens of context — well within Claude's context window while
# leaving room for the system prompt and a long answer.
MAX_CHUNKS = 15

# Characters per chunk to display in the sources summary string.
# Enough for a one-line preview in a UI without crowding.
SOURCE_PREVIEW_LEN = 120


# ─────────────────────────────────────────────────────────────────────────────
# QUERY RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    """
    Complete output of a single FinSightChain.query() call.

    Fields
    ------
    question     : str           The original user question.
    answer       : str           Claude's generated response (may be a stub if
                                 AWS credentials are not configured).
    sources      : list[str]     Unique source labels used in context, e.g.
                                 ["JPM / 10-K", "BAC / 10-Q"]. Matches the
                                 citation format embedded in the answer.
    chunks       : list[dict]    Full retrieved chunk dicts from FAISS, each with
                                 keys: text, score, ticker, filing_type, source,
                                 chunk_index, chunk_count.
    scores       : dict          BLEU, ROUGE-L, faithfulness from response_evaluator.
    ticker_filter: str | None    Ticker used to scope retrieval, or None if all
                                 companies were searched.
    is_stub      : bool          True when BedrockClient returned a stub response
                                 (no AWS call was made).
    context      : str           The full context string passed to Claude.
                                 Useful for debugging retrieval quality.
    """
    question: str
    answer: str
    sources: list[str]
    chunks: list[dict]
    scores: dict
    ticker_filter: str | None = None
    is_stub: bool = False
    context: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation. Excludes raw context to keep payloads lean."""
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "chunks": [
                {
                    "ticker": c["ticker"],
                    "filing_type": c["filing_type"],
                    "score": c["score"],
                    "chunk_index": c["chunk_index"],
                    "chunk_count": c["chunk_count"],
                    "preview": c["text"][:SOURCE_PREVIEW_LEN],
                }
                for c in self.chunks
            ],
            "scores": self.scores,
            "ticker_filter": self.ticker_filter,
            "is_stub": self.is_stub,
        }

    def __repr__(self) -> str:
        return (
            f"QueryResult("
            f"chunks={len(self.chunks)}, "
            f"sources={self.sources}, "
            f"scores={self.scores}, "
            f"stub={self.is_stub})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FINSIGHT CHAIN
# ─────────────────────────────────────────────────────────────────────────────

class FinSightChain:
    """
    End-to-end RAG pipeline for financial compliance Q&A.

    Instantiate once at application startup and reuse across all requests.
    Thread safety: FAISSEmbedder and BedrockClient are safe to call concurrently
    for reads (FAISS search, boto3 HTTP calls). The chain itself holds no
    mutable per-request state — all state lives in the returned QueryResult.
    """

    def __init__(
        self,
        embedder: FAISSEmbedder | None = None,
        bedrock: BedrockClient | None = None,
        min_score: float = MIN_SIMILARITY_SCORE,
        max_chunks: int = MAX_CHUNKS,
    ) -> None:
        """
        Parameters
        ----------
        embedder   : FAISSEmbedder, optional
            Pre-built embedder instance. If None, a new one is created using
            default paths (data/faiss_index/). Pass an existing instance to
            share the FAISS index across multiple chain instances (e.g. in a
            multi-tenant API server) without loading it twice.
        bedrock    : BedrockClient, optional
            Pre-built Bedrock client. If None, a new one is created. Auto-falls
            back to stub mode when AWS credentials are not configured.
        min_score  : float
            Cosine similarity threshold; retrieved chunks below this score are
            discarded before building the context string.
        max_chunks : int
            Hard upper bound on chunks passed to the model, applied after
            min_score filtering and optional ticker scoping.
        """
        self.min_score = min_score
        self.max_chunks = max_chunks

        # The embedder loads ~400 MB of model weights on first instantiation.
        # If the caller passed a shared instance, skip the load entirely.
        self.embedder = embedder if embedder is not None else FAISSEmbedder()

        # BedrockClient resolves AWS credentials at init time and falls back
        # to stub mode if none are found.
        self.bedrock = bedrock if bedrock is not None else BedrockClient()

        print(
            f"FinSightChain ready  "
            f"vectors={self.embedder.vector_count:,}  "
            f"min_score={min_score}  "
            f"max_chunks={max_chunks}"
        )

    # ── Primary interface ─────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: int = 5,
        ticker: str | None = None,
    ) -> QueryResult:
        """
        Run the full RAG pipeline for a single compliance question.

        Steps
        -----
        1. Embed the question and search the FAISS index (retrieve top_k × 3
           candidates to leave room for filtering).
        2. Optionally narrow results to a single company ticker.
        3. Apply the minimum similarity threshold and cap at max_chunks.
        4. Build a labelled context string from the surviving chunks.
        5. Render the compliance prompt via prompt_templates.
        6. Send to Claude Sonnet via BedrockClient.invoke().
        7. Score the response with response_evaluator.
        8. Return a QueryResult bundling all of the above.

        Parameters
        ----------
        question : str
            Natural-language compliance question, e.g.
            "What AML controls does JPMorgan disclose in its 10-K?"
        top_k : int
            Number of chunks to retrieve from FAISS. The pipeline fetches
            top_k * 3 candidates before filtering so that after score
            and ticker pruning it still has roughly top_k chunks available.
        ticker : str | None
            If provided (e.g. "JPM"), restrict context to chunks from that
            company's filings only. Useful when the question is company-
            specific and cross-company context would dilute relevance.

        Returns
        -------
        QueryResult
            See QueryResult dataclass for full field documentation.
        """
        if not question.strip():
            raise ValueError("question must be a non-empty string")

        # ── Step 1: Retrieve ─────────────────────────────────────────────────
        # Fetch 3× the requested top_k to create a buffer: some candidates
        # will be pruned by min_score or the ticker filter. Fetching more
        # upfront is much cheaper than a second FAISS round-trip.
        fetch_k = min(top_k * 3, self.embedder.vector_count or 1)
        raw_chunks = self.embedder.query(question, top_k=fetch_k)

        # ── Step 2: Filter ───────────────────────────────────────────────────
        chunks = self._filter_chunks(raw_chunks, ticker=ticker, top_k=top_k)

        if not chunks:
            return self._no_results_response(question, ticker)

        # ── Step 3: Build context ─────────────────────────────────────────────
        # Each chunk is prefixed with its source label in the exact format that
        # COMPLIANCE_SYSTEM_PROMPT instructs Claude to use for citations:
        #   [TICKER / FILING_TYPE / Chunk N of M]
        # This tight coupling between context formatting and system prompt means
        # Claude can copy labels verbatim instead of inventing citation formats.
        context = self._build_context(chunks)

        # ── Step 4: Build prompt ─────────────────────────────────────────────
        user_message = build_compliance_prompt(context=context, question=question)

        # ── Step 5: Generate ─────────────────────────────────────────────────
        response: InvokeResponse = self.bedrock.invoke(
            prompt=user_message,
            system_prompt=COMPLIANCE_SYSTEM_PROMPT,
        )

        # ── Step 6: Evaluate ─────────────────────────────────────────────────
        # Pass the raw context (not the labelled version) to the evaluator so
        # the source-label prefixes don't artificially inflate BLEU/ROUGE scores
        # by adding shared tokens that aren't present in the answer.
        raw_context_text = " ".join(c["text"] for c in chunks)
        eval_result: EvaluationResult = evaluate_response(
            answer=response.text,
            context=raw_context_text,
            question=question,
        )

        # ── Step 7: Assemble result ───────────────────────────────────────────
        sources = _unique_sources(chunks)

        return QueryResult(
            question=question,
            answer=response.text,
            sources=sources,
            chunks=chunks,
            scores=eval_result.to_dict(),
            ticker_filter=ticker,
            is_stub=response.is_stub,
            context=context,
        )

    # ── Batch interface ───────────────────────────────────────────────────────

    def batch_query(
        self,
        questions: list[str],
        top_k: int = 5,
        ticker: str | None = None,
    ) -> list[QueryResult]:
        """
        Run query() for each question sequentially and return all results.

        Sequential rather than concurrent because BedrockClient's boto3 session
        is not thread-safe for concurrent writes. For parallel execution, create
        one FinSightChain per worker thread (each gets its own boto3 session).

        Parameters
        ----------
        questions : list[str]   One or more compliance questions.
        top_k     : int         Forwarded to query() for each question.
        ticker    : str | None  Optional company scope applied to all questions.

        Returns
        -------
        list[QueryResult]  One result per question, in the same order as input.
        """
        return [self.query(q, top_k=top_k, ticker=ticker) for q in questions]

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """
        Return a health-check snapshot of the chain's components.

        Useful for API /health endpoints and startup validation — a chain with
        zero vectors in the index will always return empty results.
        """
        return {
            "vector_count": self.embedder.vector_count,
            "index_ready": self.embedder.vector_count > 0,
            "bedrock_stub": self.bedrock._stub_mode,
            "min_score": self.min_score,
            "max_chunks": self.max_chunks,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _filter_chunks(
        self,
        chunks: list[dict],
        ticker: str | None,
        top_k: int,
    ) -> list[dict]:
        """
        Apply ticker filter, minimum similarity threshold, and top_k cap.

        The order of operations matters:
          1. Ticker filter first — removes entire companies, may leave few chunks.
          2. Min-score filter second — removes low-quality matches.
          3. Top-k cap last — keeps the best remaining candidates.

        This ordering ensures the ticker filter doesn't accidentally promote
        low-score chunks from the target company over high-score chunks that
        were discarded by the ticker constraint.
        """
        if ticker:
            # Case-insensitive match to handle "jpm" vs "JPM" from the caller.
            upper = ticker.upper()
            chunks = [c for c in chunks if c.get("ticker", "").upper() == upper]

        # Discard chunks below the similarity floor.
        chunks = [c for c in chunks if c.get("score", 0.0) >= self.min_score]

        # FAISS already returns chunks sorted by score descending; slicing
        # directly preserves that ordering.
        return chunks[:min(top_k, self.max_chunks)]

    @staticmethod
    def _build_context(chunks: list[dict]) -> str:
        """
        Format retrieved chunks into a labelled context string for Claude.

        Each chunk is prefixed with its source label in the format that
        COMPLIANCE_SYSTEM_PROMPT prescribes for inline citations, so Claude
        can reference them verbatim. A blank line separates chunks so the
        model can distinguish where one passage ends and another begins.

        Example output:
            [JPM / 10-K / Chunk 4 of 31]
            JPMorgan maintains AML controls under the Bank Secrecy Act...

            [BAC / 10-Q / Chunk 7 of 28]
            Bank of America's AML program includes transaction monitoring...
        """
        parts = []
        for chunk in chunks:
            label = (
                f"[{chunk.get('ticker', 'UNKNOWN')} / "
                f"{chunk.get('filing_type', 'UNKNOWN')} / "
                f"Chunk {chunk.get('chunk_index', 0) + 1} of "
                f"{chunk.get('chunk_count', '?')}]"
            )
            parts.append(f"{label}\n{chunk['text']}")
        return "\n\n".join(parts)

    def _no_results_response(
        self,
        question: str,
        ticker: str | None,
    ) -> QueryResult:
        """
        Return a graceful QueryResult when retrieval yields no usable chunks.

        Rather than raising an exception (which would crash the API caller),
        we return a deterministic "no results" answer that the caller can
        display or log. This handles three legitimate scenarios:
          • The FAISS index is empty (no filings ingested yet).
          • The ticker filter left no results (company not in corpus).
          • All candidate chunks fell below MIN_SIMILARITY_SCORE.
        """
        if self.embedder.vector_count == 0:
            reason = (
                "The document index is empty. Run the ingestion pipeline "
                "(sec_loader → chunker → embedder) to load SEC filings."
            )
        elif ticker:
            reason = (
                f"No filings for ticker '{ticker}' were found in the index "
                f"with a similarity score ≥ {self.min_score}. "
                f"Check that '{ticker}' was included in the ingestion run."
            )
        else:
            reason = (
                f"No document chunks scored above the minimum similarity "
                f"threshold ({self.min_score}) for this question. "
                "Try rephrasing or broadening the query."
            )

        return QueryResult(
            question=question,
            answer=f"No relevant context found. {reason}",
            sources=[],
            chunks=[],
            scores={"bleu": None, "rouge_l": None, "faithfulness": None,
                    "overall": "UNKNOWN", "available_metrics": []},
            ticker_filter=ticker,
            is_stub=False,
            context="",
        )


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _unique_sources(chunks: list[dict]) -> list[str]:
    """
    Deduplicated list of 'TICKER / FILING_TYPE' strings, in retrieval order.

    Preserves the order in which sources first appear so the most relevant
    source is listed first — useful for UI displays where only the top N
    sources are shown.
    """
    seen: set[str] = set()
    sources: list[str] = []
    for chunk in chunks:
        label = f"{chunk.get('ticker', 'UNKNOWN')} / {chunk.get('filing_type', 'UNKNOWN')}"
        if label not in seen:
            seen.add(label)
            sources.append(label)
    return sources


# ─────────────────────────────────────────────────────────────────────────────
# DEMO / CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rag_pipeline.ingestion.chunker import chunk_documents
    from rag_pipeline.ingestion.sec_loader import load_filings

    # ── Bootstrap the index with stub filings (no SEC network access needed) ──
    print("Loading stub filings and building FAISS index...")
    docs = load_filings(force_stubs=True)
    chunks = chunk_documents(docs)

    embedder = FAISSEmbedder()
    if embedder.vector_count == 0:
        embedder.upsert(chunks)

    # ── Build chain ───────────────────────────────────────────────────────────
    chain = FinSightChain(embedder=embedder)
    print(f"\nChain status: {chain.status()}")

    # ── Run queries ───────────────────────────────────────────────────────────
    queries = [
        ("What are the AML and BSA compliance obligations for financial institutions?", None),
        ("How does JPMorgan manage suspicious activity report filings?", "JPM"),
        ("What capital requirements do banks face under Basel III?", None),
    ]

    for question, ticker in queries:
        scope = f"  (ticker={ticker})" if ticker else ""
        print(f"\n{'─' * 60}")
        print(f"Q: {question}{scope}")

        result = chain.query(question, top_k=5, ticker=ticker)

        print(f"Sources : {result.sources}")
        print(f"Chunks  : {len(result.chunks)} retrieved  "
              f"(best score={result.chunks[0]['score']:.3f})" if result.chunks else "0 retrieved")
        print(f"Scores  : {result.scores}")
        print(f"Stub    : {result.is_stub}")
        print(f"\nAnswer:\n{textwrap.fill(result.answer[:600], width=72)}"
              f"{'...' if len(result.answer) > 600 else ''}")

    # ── Batch query demo ──────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("Batch query demo (3 questions)...")
    batch_results = chain.batch_query(
        [
            "What fraud detection controls do banks disclose?",
            "Describe OCC consent order disclosures for AML deficiencies.",
            "What are FINRA settlement obligations for financial firms?",
        ],
        top_k=3,
    )
    for r in batch_results:
        print(f"  [{r.scores.get('overall', '?')}] {r.question[:60]}  "
              f"sources={r.sources}")
