"""
RAG compliance Q&A endpoint for the FinSight API.

The caller posts a natural-language question (and an optional ticker to scope
results to one company) and receives a grounded answer with source citations,
retrieved chunk summaries, and automated quality scores.

Why we cache the chain
-----------------------
FinSightChain is expensive to initialise:
  - FAISSEmbedder loads ~400 MB of SentenceTransformer model weights and reads
    the FAISS index from disk — ~2–4 s on first call.
  - BedrockClient creates a boto3 session and resolves AWS credentials — ~50 ms.

If the chain were created inside the request handler it would re-pay this cost
on every API call, turning a <500 ms inference into a 4+ second round-trip.

Two mechanisms cooperate to ensure the chain is loaded exactly once:

  1. LIFESPAN (preferred in production)
     api/main.py loads the chain during FastAPI startup and stores it on
     `app.state.chain`. The endpoint reads `request.app.state.chain` first.
     This is the cleanest approach: the chain is created in a controlled,
     observable startup step and its lifecycle is tied to the server process.

  2. @lru_cache FALLBACK
     `_get_chain()` is decorated with @lru_cache(maxsize=1). If `app.state`
     doesn't have a chain (e.g. in unit tests that call the router function
     directly, or in scripts that use the router outside a full FastAPI app),
     `_get_chain()` creates one and caches it for the lifetime of the process.
     Subsequent calls to `_get_chain()` return the same object instantly from
     Python's function cache without touching the filesystem or network.

     @lru_cache vs a module-level global
     Both create a singleton. The difference is explicitness and testability:
     a module-level `_chain = FinSightChain()` runs at import time — during
     linting, test collection, and CLI --help invocations — loading 400 MB of
     model weights whether or not the chain will ever be used. @lru_cache is
     lazy: the chain is only created when the first actual request arrives.
     Tests can also call `_get_chain.cache_clear()` to force a fresh instance
     without monkey-patching module state.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# CHAIN ACCESSOR
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_chain():
    """
    Create and cache a FinSightChain singleton.

    Called only when `request.app.state.chain` is unavailable (see module
    docstring). The lru_cache decorator ensures this function body runs at
    most once per process — every subsequent call returns the cached instance
    without re-loading model weights or re-reading the FAISS index.

    Cache invalidation:
        from api.routers.query import _get_chain
        _get_chain.cache_clear()   # forces a fresh instance on next call

    This is useful in integration tests that swap the FAISS index between
    test cases — clear the cache, then the next request picks up the new index.
    """
    from rag_pipeline.chain import FinSightChain
    return FinSightChain()


def _resolve_chain(request: Request):
    """
    Return the chain from app.state (lifespan) or fall back to the lru_cache.

    The fallback path is a safety net, not the primary path. In production the
    lifespan always populates app.state.chain before any request is served.
    """
    chain = getattr(request.app.state, "chain", None)
    if chain is not None:
        return chain
    return _get_chain()


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST MODEL
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """
    Compliance Q&A query.

    question : Natural-language question about SEC filings, AML obligations,
               capital ratios, fraud disclosures, or any regulatory topic
               covered by the ingested corpus.
    ticker   : Optional UPPERCASE company ticker (e.g. "JPM", "BAC") to
               restrict context to a single company's filings. When absent,
               the retriever searches the entire corpus.
    top_k    : Number of document chunks to retrieve. More chunks give the
               model more evidence but also increase token usage and latency.
               5 is the recommended default for single-question Q&A.
    """
    question: str = Field(
        ...,
        min_length  = 5,
        max_length  = 2000,
        description = "Natural-language compliance question.",
        examples    = ["What are JPMorgan's AML obligations under the Bank Secrecy Act?"],
    )
    ticker: str | None = Field(
        default     = None,
        description = "Restrict retrieval to this company's filings (e.g. 'JPM').",
        examples    = ["JPM"],
    )
    top_k: int = Field(
        default     = 5,
        ge          = 1,
        le          = 15,
        description = "Number of document chunks to retrieve (1–15). Default 5.",
    )

    @field_validator("ticker")
    @classmethod
    def normalise_ticker(cls, v: str | None) -> str | None:
        """Accept lowercase tickers from callers; normalise to uppercase internally."""
        return v.upper().strip() if v else None

    model_config = {"json_schema_extra": {
        "example": {
            "question": "What AML controls does JPMorgan maintain under the Bank Secrecy Act?",
            "ticker":   "JPM",
            "top_k":    5,
        }
    }}


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────────

class ChunkSummary(BaseModel):
    """
    Abbreviated representation of one retrieved document chunk.

    Full chunk text is omitted from the response to keep payload sizes
    manageable — the answer already incorporates the relevant content.
    Callers that need the raw text can use the source label to look up
    the filing in the ingested corpus.
    """
    ticker:      str   = Field(description="Company ticker, e.g. 'JPM'.")
    filing_type: str   = Field(description="SEC filing type: '10-K', '10-Q', or '8-K'.")
    score:       float = Field(description="Cosine similarity score (0–1). Higher = more relevant.")
    chunk_index: int   = Field(description="0-based position of this chunk within its source document.")
    chunk_count: int   = Field(description="Total chunks in the source document.")
    preview:     str   = Field(description="First 120 characters of the chunk text.")

    @classmethod
    def from_chunk(cls, chunk: dict[str, Any]) -> "ChunkSummary":
        return cls(
            ticker      = chunk.get("ticker", ""),
            filing_type = chunk.get("filing_type", ""),
            score       = round(chunk.get("score", 0.0), 4),
            chunk_index = chunk.get("chunk_index", 0),
            chunk_count = chunk.get("chunk_count", 0),
            preview     = chunk.get("text", "")[:120].strip(),
        )


class EvaluationScores(BaseModel):
    """
    Automated quality scores for the generated answer.

    These measure how well the answer is grounded in the retrieved context,
    not whether the answer is factually correct. A high faithfulness score
    means nearly all sentences in the answer are supported by retrieved
    passages — a low score may indicate hallucination.

    All scores are None when the required library (nltk / rouge-score) is not
    installed. Install both for full evaluation:
        pip install nltk rouge-score
    """
    bleu: float | None = Field(
        description=(
            "4-gram BLEU score vs. retrieved context. "
            "≥0.25 = good lexical alignment. None if nltk unavailable."
        ),
    )
    rouge_l: float | None = Field(
        description=(
            "ROUGE-L F1 (longest common subsequence) vs. retrieved context. "
            "≥0.30 = strong structural alignment. None if rouge-score unavailable."
        ),
    )
    faithfulness: float | None = Field(
        description=(
            "Fraction of answer sentences with ≥25%% content-word overlap with "
            "retrieved context. ≥0.80 = highly grounded. None if nltk unavailable."
        ),
    )
    overall: str = Field(
        description="Aggregate verdict: PASS / WARN / FAIL / UNKNOWN.",
        examples=["PASS"],
    )
    available_metrics: list[str] = Field(
        description="Which metrics were actually computed (libraries installed).",
    )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvaluationScores":
        return cls(
            bleu               = d.get("bleu"),
            rouge_l            = d.get("rouge_l"),
            faithfulness       = d.get("faithfulness"),
            overall            = d.get("overall", "UNKNOWN"),
            available_metrics  = d.get("available_metrics", []),
        )


class QueryResponse(BaseModel):
    """Compliance Q&A result including the generated answer and retrieval metadata."""

    question: str = Field(description="The original question, echoed back for traceability.")
    answer:   str = Field(description="Claude's grounded response with inline source citations.")
    sources:  list[str] = Field(
        description=(
            "Unique source labels for the documents cited in the answer, "
            "in order of relevance. Format: 'TICKER / FILING_TYPE'."
        ),
        examples=[["JPM / 10-K", "JPM / 10-Q"]],
    )
    chunks_retrieved: int = Field(
        description="Number of document chunks that passed retrieval and scoring filters.",
    )
    top_chunks: list[ChunkSummary] = Field(
        description="Summarised metadata for the top retrieved chunks (up to top_k).",
    )
    scores: EvaluationScores = Field(
        description="Automated groundedness scores for the generated answer.",
    )
    ticker_filter: str | None = Field(
        description="Ticker used to scope retrieval, or null if the full corpus was searched.",
    )
    is_stub: bool = Field(
        description=(
            "True when BedrockClient is in stub mode (no AWS credentials configured). "
            "The answer is a canned placeholder, not a real Claude response."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model = QueryResponse,
    status_code    = status.HTTP_200_OK,
    summary        = "Ask a compliance question",
    description    = (
        "Retrieves relevant SEC filing passages, sends them to Claude as grounded "
        "context, and returns a cited answer with automated quality scores. "
        "Optionally filter by company ticker to scope the search."
    ),
)
async def compliance_query(
    request: Request,
    body:    QueryRequest,
) -> QueryResponse:
    """
    Run the full RAG pipeline for a compliance question.

    Pipeline steps (see rag_pipeline/chain.py for detail):
      1. Embed the question with SentenceTransformer (768-dim vector).
      2. Search FAISS index for the top-k most similar SEC filing chunks.
      3. Optionally narrow results to the requested ticker.
      4. Format chunks as labelled context ([TICKER / TYPE / Chunk N of M]).
      5. Send context + question to Claude Sonnet via AWS Bedrock.
      6. Score the response with BLEU, ROUGE-L, and faithfulness metrics.
      7. Return the answer, source labels, chunk metadata, and quality scores.
    """
    # Resolve the chain — prefers app.state (lifespan) over lru_cache fallback.
    chain = _resolve_chain(request)
    if chain is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = (
                "RAG chain is not available. The server may still be loading the "
                "FAISS index and embedding model. Check GET /health/ready."
            ),
        )

    # ── Run the pipeline ───────────────────────────────────────────────────────
    try:
        result = chain.query(
            question = body.question,
            top_k    = body.top_k,
            ticker   = body.ticker,
        )
    except Exception as exc:
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = f"RAG pipeline failed: {exc}",
        ) from exc

    # ── Assemble response ──────────────────────────────────────────────────────
    return QueryResponse(
        question         = result.question,
        answer           = result.answer,
        sources          = result.sources,
        chunks_retrieved = len(result.chunks),
        top_chunks       = [ChunkSummary.from_chunk(c) for c in result.chunks],
        scores           = EvaluationScores.from_dict(result.scores),
        ticker_filter    = result.ticker_filter,
        is_stub          = result.is_stub,
    )
