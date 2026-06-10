"""
FinSight Agentic RAG pipeline — LangGraph-powered multi-step retrieval agent.

What LangGraph is
-----------------
LangGraph is a library built on top of LangChain that lets you describe
multi-step LLM workflows as a *directed graph* of nodes and edges. Each node
is a plain Python function that reads from and writes to a shared state dict.
Edges connect nodes; conditional edges let the graph branch based on runtime
values (e.g. "was the answer good enough?").

Under the hood, LangGraph compiles the graph into a state machine and runs it
step by step, passing the accumulated state forward at each hop. You get full
visibility into every intermediate result — rewritten query, retrieved chunks,
evaluation score — without tangled callback chains or opaque LLM orchestration.

Why agentic RAG is better than simple RAG
------------------------------------------
Simple RAG (like FinSightChain in chain.py) runs a fixed three-step sequence:
  retrieve → generate → evaluate
and returns whatever it gets on the first attempt, good or bad.

Agentic RAG adds *autonomy*: the pipeline can inspect its own outputs and
decide to try again with a better strategy. The improvement comes from two
sources:

1. QUERY REWRITING — The user's raw question is often a poor retrieval signal.
   "What did JPM say about AML?" is vague. An LLM-rewritten version —
   "JPMorgan Chase anti-money laundering Bank Secrecy Act suspicious activity
   reports 10-K compliance disclosure" — is dense with terminology that matches
   SEC filing language, yielding higher-scoring FAISS results.

2. SELF-CORRECTION LOOP — If the evaluator flags a low faithfulness score
   (≥ 50% of answer sentences unsupported by retrieved text), the agent rewrites
   the query from a different angle and retries retrieval + generation. This
   catches cases where the first rewrite retrieved the wrong passages and the
   generated answer compensated with hallucinated content.

The graph for this pipeline:

  ┌─────────────────────────────────────────────────────────┐
  │                                                          │
  │   [START]                                                │
  │      │                                                   │
  │      ▼                                                   │
  │  query_rewriter  ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐                │
  │      │                                 │ needs_retry=True│
  │      ▼                                 │ attempts < 2    │
  │   retriever                            │                 │
  │      │                                 │                 │
  │      ▼                                 │                 │
  │   generator                            │                 │
  │      │                                 │                 │
  │      ▼                                 │                 │
  │   evaluator                            │                 │
  │      │                                 │                 │
  │      ▼                                 │                 │
  │    router ─ retry? ────────────────────┘                 │
  │      │                                                   │
  │      └─ done? ──► [END]                                  │
  │                                                          │
  └─────────────────────────────────────────────────────────┘

State fields
------------
question         : str            The original user question (never modified).
rewritten_query  : str            The LLM-rewritten retrieval query (updated each
                                  attempt so the retry uses a fresh angle).
retrieved_chunks : list[dict]     FAISS results for the current rewritten_query.
answer           : str            Claude's generated answer for the current attempt.
evaluation_score : float          Faithfulness score from response_evaluator [0, 1].
needs_retry      : bool           Set True by evaluator when score < RETRY_THRESHOLD.
attempts         : int            Incremented each time query_rewriter runs.

Usage
-----
    from rag_pipeline.agentic_chain import AgenticRAGChain

    agent = AgenticRAGChain()                # load once at startup
    result = agent.run("What are JPMorgan's AML obligations under the BSA?")

    print(result["answer"])
    print(result["rewritten_query"])   # see what the agent actually searched for
    print(result["evaluation_score"])  # faithfulness of the final answer
    print(result["sources"])           # deduplicated [TICKER / FILING_TYPE] labels
"""

from __future__ import annotations

import textwrap
from typing import Any

# ── LangGraph ─────────────────────────────────────────────────────────────────
# StateGraph is the core LangGraph primitive. You describe the graph structure
# (nodes + edges) on a StateGraph instance, then call .compile() to get a
# runnable Pregel executor.
#
# TypedDict is used to define the state schema: LangGraph reads the annotations
# to know which keys exist and what types they carry. Any node function may
# return a partial dict — only the keys it returns are updated; all other keys
# carry forward unchanged from the previous step.
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from rag_pipeline.generation.bedrock_client import BedrockClient
from rag_pipeline.generation.prompt_templates import (
    COMPLIANCE_SYSTEM_PROMPT,
    build_compliance_prompt,
)
from rag_pipeline.generation.response_evaluator import evaluate_response
from rag_pipeline.ingestion.embedder import FAISSEmbedder

# ── Configuration ─────────────────────────────────────────────────────────────

# Faithfulness score below which the evaluator node sets needs_retry=True.
# 0.5 means fewer than half the answer sentences are grounded in retrieved text —
# a signal that the retrieved chunks were poorly matched and the model likely
# filled gaps with its training knowledge.
RETRY_THRESHOLD = 0.5

# Hard cap on self-correction iterations. Each attempt costs one rewrite call
# + one retrieval + one generation call, so keep this low. Two retries means
# up to three total attempts (original + 2 retries) before the agent gives up
# and returns the best answer it has.
MAX_ATTEMPTS = 2

# Number of FAISS neighbours to fetch per retrieval attempt.
RETRIEVAL_TOP_K = 5

# Cosine similarity floor — chunks below this score are dropped before context
# assembly even if they are in the top-k results.
MIN_SIMILARITY = 0.30


# ─────────────────────────────────────────────────────────────────────────────
# STATE SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    The shared state dict that flows through every node in the graph.

    LangGraph passes this dict into each node function and merges the returned
    partial dict back. Fields not returned by a node are unchanged. Using a
    TypedDict (rather than a plain dict) gives IDE type-checking and makes the
    state contract explicit.
    """
    question:         str             # original user question — read-only after init
    rewritten_query:  str             # LLM-rewritten query; updated on each attempt
    retrieved_chunks: list[dict]      # FAISS chunks for the current rewritten_query
    answer:           str             # Claude's answer for the current attempt
    evaluation_score: float           # faithfulness score ∈ [0, 1]
    needs_retry:      bool            # True → router sends flow back to query_rewriter
    attempts:         int             # how many times query_rewriter has run


# ─────────────────────────────────────────────────────────────────────────────
# AGENTIC RAG CHAIN
# ─────────────────────────────────────────────────────────────────────────────

class AgenticRAGChain:
    """
    LangGraph-powered agentic RAG pipeline for financial compliance Q&A.

    The graph compiles once at __init__ time. Each call to run() creates a
    fresh initial state, then executes the compiled graph from START to END,
    returning the final accumulated state.

    Instantiate once and reuse — the heavy objects (SentenceTransformer,
    FAISS index, boto3 session) are loaded at __init__ and shared across calls.
    """

    def __init__(
        self,
        embedder:      FAISSEmbedder | None = None,
        bedrock:       BedrockClient | None = None,
        retry_threshold: float = RETRY_THRESHOLD,
        max_attempts:    int   = MAX_ATTEMPTS,
        top_k:           int   = RETRIEVAL_TOP_K,
        min_similarity:  float = MIN_SIMILARITY,
    ) -> None:
        self.retry_threshold = retry_threshold
        self.max_attempts    = max_attempts
        self.top_k           = top_k
        self.min_similarity  = min_similarity

        # Shared infrastructure — expensive to load, cheap to reuse.
        self.embedder = embedder if embedder is not None else FAISSEmbedder()
        self.bedrock  = bedrock  if bedrock  is not None else BedrockClient()

        # Compile the LangGraph graph once. compile() validates the graph
        # topology (no dangling edges, START/END reachable) and returns a
        # Pregel executor that can be invoked repeatedly.
        self._graph = self._build_graph()

        print(
            f"AgenticRAGChain ready  "
            f"vectors={self.embedder.vector_count:,}  "
            f"retry_threshold={retry_threshold}  "
            f"max_attempts={max_attempts}"
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, question: str) -> dict[str, Any]:
        """
        Run the agentic RAG pipeline for a single compliance question.

        The graph may iterate up to (max_attempts + 1) times if each generated
        answer scores below the faithfulness threshold. The final state — from
        whichever attempt produced the best answer — is returned.

        Parameters
        ----------
        question : str
            Natural-language compliance question.

        Returns
        -------
        dict with keys:
            answer           : str           Final answer text from Claude.
            rewritten_query  : str           The retrieval query used in the last attempt.
            sources          : list[str]     Deduplicated "TICKER / FILING_TYPE" labels.
            chunks           : list[dict]    Full FAISS result dicts (text, score, …).
            evaluation_score : float         Faithfulness score of the final answer.
            attempts         : int           How many query-rewrite/retrieve/generate
                                             cycles ran before the agent stopped.
            needs_retry      : bool          True if the agent hit max_attempts while
                                             still below the faithfulness threshold.
        """
        if not question.strip():
            raise ValueError("question must be a non-empty string")

        # Seed the initial state. LangGraph carries all keys forward through
        # every node, updating only the keys each node explicitly returns.
        initial_state: AgentState = {
            "question":         question,
            "rewritten_query":  "",
            "retrieved_chunks": [],
            "answer":           "",
            "evaluation_score": 0.0,
            "needs_retry":      False,
            "attempts":         0,
        }

        # invoke() runs the compiled graph to completion and returns the final
        # state. Each node mutates a copy of the state dict; the final dict
        # reflects all accumulated changes across all hops.
        final_state: AgentState = self._graph.invoke(initial_state)

        # Build the sources list from whichever chunks the last attempt retrieved.
        sources = _unique_sources(final_state["retrieved_chunks"])

        return {
            "answer":           final_state["answer"],
            "rewritten_query":  final_state["rewritten_query"],
            "sources":          sources,
            "chunks":           final_state["retrieved_chunks"],
            "evaluation_score": final_state["evaluation_score"],
            "attempts":         final_state["attempts"],
            "needs_retry":      final_state["needs_retry"],
        }

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self):
        """
        Declare the node/edge topology and compile to a runnable executor.

        Nodes are added by name; edges connect them. add_conditional_edges()
        inspects the return value of the router function at runtime to decide
        which outgoing edge to follow.
        """
        graph = StateGraph(AgentState)

        # Register nodes — each node is a method on this instance so it has
        # access to self.embedder, self.bedrock, etc. without globals.
        graph.add_node("query_rewriter", self._node_query_rewriter)
        graph.add_node("retriever",      self._node_retriever)
        graph.add_node("generator",      self._node_generator)
        graph.add_node("evaluator",      self._node_evaluator)
        graph.add_node("router",         self._node_router)

        # Linear edges: the happy path flows straight through each node in order.
        graph.add_edge(START,            "query_rewriter")
        graph.add_edge("query_rewriter", "retriever")
        graph.add_edge("retriever",      "generator")
        graph.add_edge("generator",      "evaluator")
        graph.add_edge("evaluator",      "router")

        # Conditional edge from router: _route_decision() returns either the
        # string "query_rewriter" (retry) or END (finish). LangGraph maps the
        # return value to the corresponding outgoing edge.
        graph.add_conditional_edges(
            "router",
            _route_decision,
            {
                "query_rewriter": "query_rewriter",
                END:              END,
            },
        )

        return graph.compile()

    # ── Node: query_rewriter ──────────────────────────────────────────────────

    def _node_query_rewriter(self, state: AgentState) -> dict:
        """
        Rewrite the user's question into a dense retrieval query using Claude.

        Why rewriting helps
        -------------------
        User questions are written for humans; FAISS retrieval works best with
        queries that share vocabulary with the indexed documents. SEC filings
        are dense with regulatory jargon ("Bank Secrecy Act", "SAR", "CET1")
        that rarely appears verbatim in a casual question.

        On the first attempt (attempts == 0), Claude expands the question with
        relevant regulatory terminology. On retry attempts, the prompt
        explicitly asks for a *different angle* — preventing the agent from
        re-fetching the same poor chunks that triggered the retry.

        The rewritten query is stored in state["rewritten_query"] and used
        by the retriever node on this and all subsequent attempts.
        """
        attempt = state["attempts"]

        if attempt == 0:
            # First pass: expand the question into retrieval-optimised terminology.
            rewrite_prompt = (
                "You are a financial document retrieval specialist. Rewrite the "
                "following question as a dense keyword query optimised for "
                "semantic search over SEC filings (10-K, 10-Q). Include relevant "
                "regulatory terminology, acronyms, and company names. Output only "
                "the rewritten query — no explanation, no preamble.\n\n"
                f"Question: {state['question']}"
            )
        else:
            # Retry: the previous retrieval + generation cycle produced a
            # low-faithfulness answer. Ask for a different retrieval angle to
            # surface different chunks.
            rewrite_prompt = (
                "You are a financial document retrieval specialist. The previous "
                "retrieval attempt for the question below produced an answer with "
                "low faithfulness to the retrieved documents. Rewrite the question "
                "using a *different angle* — different terminology, synonyms, or a "
                "more specific sub-question — to retrieve different document passages.\n\n"
                f"Original question: {state['question']}\n"
                f"Previous query used: {state['rewritten_query']}\n\n"
                "Output only the new rewritten query — no explanation, no preamble."
            )

        response = self.bedrock.invoke(
            prompt        = rewrite_prompt,
            system_prompt = "",
            max_tokens    = 128,
            temperature   = 0.3,  # slight creativity for retry rewrites
        )

        rewritten = response.text.strip()

        # Fall back to the original question if the model returns an empty
        # string (can happen in stub mode or on token budget exhaustion).
        if not rewritten:
            rewritten = state["question"]

        print(
            f"  [query_rewriter] attempt={attempt + 1}  "
            f"query={rewritten[:80]}{'...' if len(rewritten) > 80 else ''}"
        )

        return {
            "rewritten_query": rewritten,
            "attempts":        attempt + 1,
        }

    # ── Node: retriever ───────────────────────────────────────────────────────

    def _node_retriever(self, state: AgentState) -> dict:
        """
        Search the FAISS index with the rewritten query and filter results.

        Fetches top_k × 3 candidates to create a buffer for the similarity
        floor filter — identical to the strategy in FinSightChain.query().
        The filtered, capped chunk list is stored in retrieved_chunks so
        every downstream node (generator, evaluator) sees the same set.
        """
        fetch_k    = min(self.top_k * 3, self.embedder.vector_count or 1)
        raw_chunks = self.embedder.query(state["rewritten_query"], top_k=fetch_k)

        # Apply similarity floor and top_k cap.
        chunks = [c for c in raw_chunks if c.get("score", 0.0) >= self.min_similarity]
        chunks = chunks[:self.top_k]

        best_score = chunks[0]["score"] if chunks else 0.0
        print(
            f"  [retriever] fetched={len(raw_chunks)}  "
            f"after_filter={len(chunks)}  "
            f"best_score={best_score:.3f}"
        )

        return {"retrieved_chunks": chunks}

    # ── Node: generator ───────────────────────────────────────────────────────

    def _node_generator(self, state: AgentState) -> dict:
        """
        Generate a grounded answer from the retrieved chunks using Claude.

        Context assembly and prompt rendering follow exactly the same pattern
        as FinSightChain so that citation format, system prompt constraints,
        and source labelling are consistent across both pipeline styles.
        If no chunks were retrieved, return a clear "no context" answer
        rather than letting Claude hallucinate a response.
        """
        chunks = state["retrieved_chunks"]

        if not chunks:
            return {
                "answer": (
                    "No relevant document passages were found for this query. "
                    "The index may not contain filings that address this question, "
                    "or the similarity threshold may be too high. "
                    "Try rephrasing the question or lowering the similarity floor."
                )
            }

        # Build labelled context — each chunk prefixed with its source label in
        # the [TICKER / FILING_TYPE / Chunk N of M] format that the system
        # prompt requires for inline citations.
        context_parts = []
        for chunk in chunks:
            label = (
                f"[{chunk.get('ticker', 'UNKNOWN')} / "
                f"{chunk.get('filing_type', 'UNKNOWN')} / "
                f"Chunk {chunk.get('chunk_index', 0) + 1} of "
                f"{chunk.get('chunk_count', '?')}]"
            )
            context_parts.append(f"{label}\n{chunk['text']}")
        context = "\n\n".join(context_parts)

        user_message = build_compliance_prompt(
            context  = context,
            question = state["question"],  # always the original question
        )

        response = self.bedrock.invoke(
            prompt        = user_message,
            system_prompt = COMPLIANCE_SYSTEM_PROMPT,
        )

        print(
            f"  [generator] answer_len={len(response.text)}  "
            f"stub={response.is_stub}"
        )

        return {"answer": response.text}

    # ── Node: evaluator ───────────────────────────────────────────────────────

    def _node_evaluator(self, state: AgentState) -> dict:
        """
        Score the generated answer for faithfulness and decide whether to retry.

        The faithfulness metric (from response_evaluator.py) measures the
        fraction of answer sentences that have lexical overlap with the
        retrieved context. A score below RETRY_THRESHOLD means too many
        sentences are potentially hallucinated — the chunks may not have
        contained the answer and Claude filled the gap with training knowledge.

        Setting needs_retry=True signals the router to send the flow back to
        query_rewriter for another attempt.
        """
        chunks = state["retrieved_chunks"]

        if not chunks or not state["answer"]:
            # No chunks or empty answer — can't evaluate, skip retry.
            return {"evaluation_score": 0.0, "needs_retry": False}

        raw_context = " ".join(c["text"] for c in chunks)

        eval_result = evaluate_response(
            answer   = state["answer"],
            context  = raw_context,
            question = state["question"],
        )

        # Use faithfulness as the primary retry signal: it directly measures
        # whether the answer is grounded in the retrieved passages.
        score = eval_result.faithfulness if eval_result.faithfulness is not None else 0.0
        needs_retry = score < self.retry_threshold

        print(
            f"  [evaluator] faithfulness={score:.3f}  "
            f"overall={eval_result.overall}  "
            f"needs_retry={needs_retry}"
        )

        return {
            "evaluation_score": score,
            "needs_retry":      needs_retry,
        }

    # ── Node: router ──────────────────────────────────────────────────────────

    def _node_router(self, state: AgentState) -> dict:
        """
        Log the routing decision. The actual branching is handled by the
        conditional edge function _route_decision(), which LangGraph calls
        after this node returns.

        Separating the routing logic into a standalone function (rather than
        embedding it in this node) keeps _route_decision() pure and easily
        unit-testable.
        """
        decision = _route_decision(state)
        target = "query_rewriter (retry)" if decision == "query_rewriter" else "END"
        print(
            f"  [router] attempts={state['attempts']}  "
            f"needs_retry={state['needs_retry']}  "
            f"→ {target}"
        )
        # Router node does not modify state — it is purely a logging/branching hop.
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONAL EDGE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def _route_decision(state: AgentState) -> str:
    """
    Decide the next node after the router.

    Called by LangGraph's conditional edge machinery after the router node
    returns. Must return a string that matches one of the keys in the
    add_conditional_edges() path map.

    Logic
    -----
    Retry if BOTH conditions hold:
      1. needs_retry is True (evaluator flagged low faithfulness)
      2. attempts < MAX_ATTEMPTS (we haven't exhausted the retry budget)

    When the retry budget is exhausted, the agent ends with whatever answer
    it produced on the last attempt — even if it's below the threshold.
    This prevents infinite loops while still giving the agent multiple
    chances to improve.
    """
    if state["needs_retry"] and state["attempts"] < MAX_ATTEMPTS:
        return "query_rewriter"
    return END


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _unique_sources(chunks: list[dict]) -> list[str]:
    """
    Deduplicated list of 'TICKER / FILING_TYPE' strings, in retrieval order.

    Mirrors the helper in chain.py so source formatting is consistent between
    the simple and agentic pipelines.
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

    print("Loading stub filings and building FAISS index...")
    docs   = load_filings(force_stubs=True)
    chunks = chunk_documents(docs)

    embedder = FAISSEmbedder()
    if embedder.vector_count == 0:
        embedder.upsert(chunks)

    agent = AgenticRAGChain(embedder=embedder)

    queries = [
        "What are JPMorgan's AML obligations under the Bank Secrecy Act?",
        "How do banks calculate CET1 capital ratios under Basel III?",
        "What fraud detection controls do financial institutions disclose in 10-K filings?",
    ]

    for question in queries:
        print(f"\n{'─' * 68}")
        print(f"Q: {question}")

        result = agent.run(question)

        print(f"\nRewritten query : {result['rewritten_query']}")
        print(f"Sources         : {result['sources']}")
        print(f"Chunks retrieved: {len(result['chunks'])}")
        print(f"Faithfulness    : {result['evaluation_score']:.3f}")
        print(f"Attempts        : {result['attempts']}")
        print(f"\nAnswer:\n{textwrap.fill(result['answer'][:600], width=72)}"
              f"{'...' if len(result['answer']) > 600 else ''}")
