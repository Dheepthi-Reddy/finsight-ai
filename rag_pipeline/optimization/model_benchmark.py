"""
Bedrock model benchmarking for the FinSight RAG pipeline.

Why benchmark multiple models
-------------------------------
Claude Haiku and Claude Sonnet are not interchangeable — they occupy different
points on the latency/quality/cost tradeoff curve:

  Model                                     Input $/1M   Output $/1M   Typical p50
  ─────────────────────────────────────────────────────────────────────────────────
  claude-haiku-4-5   (fast, cheap)           $0.80         $4.00        ~0.8s
  claude-sonnet-4-5  (balanced)              $3.00        $15.00        ~2.0s

For the FinSight compliance assistant, the right model depends on the query:
  • Simple factual lookup ("What is JPMorgan's CET1 ratio?") → Haiku is fine.
  • Complex multi-step regulatory analysis → Sonnet's reasoning depth matters.

Running both models on the same question with the same context lets us:
  1. Quantify the actual latency difference for our specific prompt lengths.
  2. Measure answer quality objectively with the LLMJudge scorer.
  3. Calculate cost-per-query for each model given real token counts.
  4. Build a routing heuristic: use Haiku for short answers, Sonnet for complex
     queries — maximising quality/cost without manually auditing every question.

What this benchmark measures
------------------------------
For each model × question combination:
  LATENCY     — wall-clock time from invoke() call to first byte of response.
                Measured with time.perf_counter() for sub-millisecond precision.
  INPUT TOKENS  — reported by Bedrock in the response usage field. Determines
                  the input cost component.
  OUTPUT TOKENS — reported by Bedrock. Determines the output cost component.
  COST (est.)  — input_tokens × input_price + output_tokens × output_price.
                Prices are hardcoded constants; update them if AWS changes pricing.
  QUALITY     — faithfulness, relevance, completeness from LLMJudge.judge().
                Only run when run_quality_eval=True (adds one extra Bedrock call
                per model to avoid inflating costs in routine benchmarks).

Usage
-----
    from rag_pipeline.optimization.model_benchmark import ModelBenchmark

    bench = ModelBenchmark()
    report = bench.benchmark(
        question = "What are JPMorgan's AML obligations under the BSA?",
        context  = retrieved_chunks_text,
    )
    print(report["summary"])
    print(report["recommendation"])
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rag_pipeline.generation.bedrock_client import BedrockClient, InvokeResponse
from rag_pipeline.generation.prompt_templates import (
    COMPLIANCE_SYSTEM_PROMPT,
    build_compliance_prompt,
)

# ── Model registry ─────────────────────────────────────────────────────────────
# Each entry: (model_id, display_name, input_$/1M_tokens, output_$/1M_tokens)
# Prices sourced from https://aws.amazon.com/bedrock/pricing/ — update when
# AWS revises them. Cost calculations are estimates, not billing guarantees.
MODELS = [
    (
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "Claude Haiku 4.5",
        0.80,    # $0.80 per 1M input tokens
        4.00,    # $4.00 per 1M output tokens
    ),
    (
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "Claude Sonnet 4.5",
        3.00,    # $3.00 per 1M input tokens
        15.00,   # $15.00 per 1M output tokens
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    """
    Benchmark measurements for a single model × question run.

    Fields
    ------
    model_id      : str    Bedrock model identifier.
    model_name    : str    Human-readable display name.
    answer        : str    The model's generated response.
    latency_s     : float  Wall-clock seconds from invoke() to response.
    input_tokens  : int    Tokens in the prompt (from Bedrock usage).
    output_tokens : int    Tokens generated (from Bedrock usage).
    cost_usd      : float  Estimated cost in USD for this single call.
    is_stub       : bool   True when BedrockClient ran in stub mode.
    quality       : dict   Keys: faithfulness, relevance, completeness, verdict.
                           Empty dict when quality eval was not requested.
    error         : str    Non-empty if the invocation failed; other fields
                           will be at their zero values.
    """
    model_id:      str
    model_name:    str
    answer:        str
    latency_s:     float
    input_tokens:  int
    output_tokens: int
    cost_usd:      float
    is_stub:       bool = False
    quality:       dict = field(default_factory=dict)
    error:         str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id":      self.model_id,
            "model_name":    self.model_name,
            "latency_s":     round(self.latency_s, 3),
            "input_tokens":  self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd":      round(self.cost_usd, 6),
            "is_stub":       self.is_stub,
            "quality":       self.quality,
            "error":         self.error,
            "answer_preview": self.answer[:200] + ("..." if len(self.answer) > 200 else ""),
        }

    def __repr__(self) -> str:
        quality_str = (
            f"  faith={self.quality.get('faithfulness', '?'):.2f}"
            f"  rel={self.quality.get('relevance', '?'):.2f}"
            if self.quality else ""
        )
        return (
            f"ModelResult({self.model_name}  "
            f"latency={self.latency_s:.2f}s  "
            f"tokens={self.input_tokens}+{self.output_tokens}  "
            f"cost=${self.cost_usd:.4f}"
            f"{quality_str})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MODEL BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

class ModelBenchmark:
    """
    Runs the same compliance prompt against multiple Bedrock models and
    produces a comparison report with latency, cost, and quality scores.

    Instantiate once and reuse — BedrockClient instances for each model are
    created at __init__ time and share the underlying boto3 session pool.
    """

    def __init__(
        self,
        models:           list[tuple] = MODELS,
        run_quality_eval: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        models           : list[tuple]
            List of (model_id, display_name, input_price, output_price) tuples.
            Defaults to MODELS (Haiku + Sonnet). Override to test other models
            or to add new Claude versions as they become available on Bedrock.
        run_quality_eval : bool
            Whether to run LLMJudge quality evaluation on each answer. Adds one
            extra Bedrock invoke per model. Set False for quick latency/cost
            benchmarks where answer quality is not the focus.
        """
        self._models = models
        self._run_quality_eval = run_quality_eval

        # Create one BedrockClient per model so each invocation uses the
        # correct model_id without needing to override it per call.
        # All clients share the same AWS region and credential chain.
        self._clients: dict[str, tuple[BedrockClient, str, float, float]] = {}
        for model_id, name, input_price, output_price in models:
            client = BedrockClient(model_id=model_id)
            self._clients[model_id] = (client, name, input_price, output_price)

        # Import LLMJudge lazily to avoid circular imports and to allow the
        # benchmark to run in cost-only mode without instantiating a judge.
        self._judge = None
        if run_quality_eval:
            from rag_pipeline.evaluation.llm_judge import LLMJudge
            # Use the default (first) model's client as the judge to avoid
            # inflating benchmark costs with an extra high-cost model invocation.
            judge_client = self._clients[models[0][0]][0]
            self._judge = LLMJudge(bedrock=judge_client)

        stub_modes = [c._stub_mode for c, *_ in self._clients.values()]
        print(
            f"ModelBenchmark ready  "
            f"models={len(models)}  "
            f"quality_eval={run_quality_eval}  "
            f"stub={any(stub_modes)}"
        )

    # ── benchmark ─────────────────────────────────────────────────────────────

    def benchmark(
        self,
        question:      str,
        context:       str,
        system_prompt: str = COMPLIANCE_SYSTEM_PROMPT,
        max_tokens:    int = 512,
        temperature:   float = 0.0,
    ) -> dict[str, Any]:
        """
        Run the same prompt on every registered model and return a report.

        Each model receives an identical prompt (built from the same question
        and context) so differences in the results are attributable to model
        differences, not prompt differences. Temperature is fixed at 0.0 for
        deterministic outputs — this makes latency and token counts reproducible
        across benchmark runs.

        Parameters
        ----------
        question      : str   The compliance question to answer.
        context       : str   Retrieved document chunks (same for all models).
        system_prompt : str   System prompt (defaults to COMPLIANCE_SYSTEM_PROMPT).
        max_tokens    : int   Capped at 512 for benchmarking — enough for a
                              representative answer without inflating cost.
        temperature   : float Fixed at 0.0 for determinism.

        Returns
        -------
        dict with keys:
            results        : list[ModelResult]   Per-model measurements.
            summary        : list[dict]          Tabular summary for display.
            recommendation : str                 Which model to use and why.
            fastest        : str                 model_name of the lowest-latency result.
            cheapest       : str                 model_name of the lowest-cost result.
            best_quality   : str | None          model_name with highest avg quality score,
                                                 or None if quality eval was skipped.
        """
        user_message = build_compliance_prompt(context=context, question=question)
        results: list[ModelResult] = []

        for model_id, (client, name, input_price, output_price) in self._clients.items():
            print(f"  Benchmarking {name}...")
            result = self._run_single(
                client=client,
                model_id=model_id,
                model_name=name,
                input_price=input_price,
                output_price=output_price,
                user_message=user_message,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                question=question,
                context=context,
            )
            results.append(result)
            print(f"    {result}")

        report = self._build_report(results)
        return report

    # ── batch benchmark ───────────────────────────────────────────────────────

    def benchmark_batch(
        self,
        questions: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        """
        Run benchmark() for multiple question/context pairs.

        Parameters
        ----------
        questions : list[dict]
            Each dict must have keys "question" and "context".

        Returns
        -------
        list[dict]   One report dict per question, in the same order as input.
        """
        reports = []
        for i, item in enumerate(questions):
            print(f"\nBenchmark {i + 1}/{len(questions)}: {item['question'][:60]}...")
            reports.append(self.benchmark(
                question=item["question"],
                context=item["context"],
            ))
        return reports

    # ── private helpers ───────────────────────────────────────────────────────

    def _run_single(
        self,
        client:        BedrockClient,
        model_id:      str,
        model_name:    str,
        input_price:   float,
        output_price:  float,
        user_message:  str,
        system_prompt: str,
        max_tokens:    int,
        temperature:   float,
        question:      str,
        context:       str,
    ) -> ModelResult:
        """Invoke one model, measure latency, compute cost, optionally evaluate quality."""
        try:
            t0 = time.perf_counter()
            response: InvokeResponse = client.invoke(
                prompt=user_message,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            latency_s = time.perf_counter() - t0

            cost_usd = _estimate_cost(
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                input_price=input_price,
                output_price=output_price,
            )

            quality: dict[str, Any] = {}
            if self._judge and not response.is_stub:
                judge_result = self._judge.judge(
                    question=question,
                    answer=response.text,
                    context=context,
                )
                quality = judge_result.to_dict()

            return ModelResult(
                model_id=model_id,
                model_name=model_name,
                answer=response.text,
                latency_s=latency_s,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=cost_usd,
                is_stub=response.is_stub,
                quality=quality,
            )

        except Exception as exc:
            # Return a failed result rather than propagating the exception so
            # the benchmark can still report results for the other models.
            return ModelResult(
                model_id=model_id,
                model_name=model_name,
                answer="",
                latency_s=0.0,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _build_report(self, results: list[ModelResult]) -> dict[str, Any]:
        """Aggregate per-model results into a comparison report dict."""
        # Filter out failed results for ranking comparisons.
        ok = [r for r in results if not r.error]

        fastest = min(ok, key=lambda r: r.latency_s).model_name if ok else "N/A"
        cheapest = min(ok, key=lambda r: r.cost_usd).model_name if ok else "N/A"
        best_quality = None

        if self._judge and ok:
            def _avg_quality(r: ModelResult) -> float:
                q = r.quality
                if not q:
                    return 0.0
                return (
                    q.get("faithfulness", 0.0)
                    + q.get("relevance", 0.0)
                    + q.get("completeness", 0.0)
                ) / 3

            best_quality = max(ok, key=_avg_quality).model_name

        recommendation = _build_recommendation(ok, fastest, cheapest, best_quality)

        return {
            "results":        results,
            "summary":        [r.to_dict() for r in results],
            "fastest":        fastest,
            "cheapest":       cheapest,
            "best_quality":   best_quality,
            "recommendation": recommendation,
        }


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_cost(
    input_tokens:  int,
    output_tokens: int,
    input_price:   float,
    output_price:  float,
) -> float:
    """
    Estimate cost in USD for one Bedrock invocation.

    Prices are per 1M tokens; convert to per-token by dividing by 1,000,000.
    This is an estimate — actual billing rounds to the nearest 1,000 tokens
    and may include request overhead charges depending on the AWS pricing tier.
    """
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def _build_recommendation(
    results:      list[ModelResult],
    fastest:      str,
    cheapest:     str,
    best_quality: str | None,
) -> str:
    """
    Produce a plain-English recommendation based on the benchmark results.

    The recommendation favours the model that offers the best quality-per-dollar
    tradeoff. If quality evaluation was skipped, it falls back to recommending
    the fastest model for latency-sensitive use cases and the cheapest for
    bulk processing.
    """
    if not results:
        return "No results available — all models failed."

    if best_quality:
        # Quality data available: recommend based on quality/cost tradeoff.
        quality_winner = next(
            (r for r in results if r.model_name == best_quality), None
        )
        cheapest_model = next(
            (r for r in results if r.model_name == cheapest), None
        )

        if quality_winner and cheapest_model:
            quality_diff = (
                quality_winner.quality.get("faithfulness", 0)
                + quality_winner.quality.get("relevance", 0)
                + quality_winner.quality.get("completeness", 0)
            ) / 3 - (
                cheapest_model.quality.get("faithfulness", 0)
                + cheapest_model.quality.get("relevance", 0)
                + cheapest_model.quality.get("completeness", 0)
            ) / 3

            cost_ratio = (
                quality_winner.cost_usd / cheapest_model.cost_usd
                if cheapest_model.cost_usd > 0 else 1.0
            )

            if quality_diff < 0.05 and cost_ratio > 1.5:
                return (
                    f"Use {cheapest} — quality difference is negligible "
                    f"({quality_diff:+.2f}) but it is {cost_ratio:.1f}× cheaper."
                )
            else:
                return (
                    f"Use {best_quality} — superior quality score "
                    f"(+{quality_diff:.2f} avg) justifies the higher cost."
                )

    # No quality data — recommend based on latency and cost.
    if fastest == cheapest:
        return f"Use {fastest} — it is both the fastest and the cheapest option."

    latency_winner = next((r for r in results if r.model_name == fastest), None)
    cost_winner = next((r for r in results if r.model_name == cheapest), None)

    if latency_winner and cost_winner:
        latency_diff = cost_winner.latency_s - latency_winner.latency_s
        cost_savings = latency_winner.cost_usd - cost_winner.cost_usd
        return (
            f"Use {fastest} for latency-sensitive queries "
            f"({latency_diff:.2f}s faster). "
            f"Use {cheapest} for bulk processing "
            f"(${cost_savings:.5f} cheaper per query)."
        )

    return f"Use {fastest} — fastest model in the benchmark."


# ─────────────────────────────────────────────────────────────────────────────
# DEMO / CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CONTEXT = (
        "JPMorgan Chase maintains comprehensive AML compliance programs governed "
        "by the Bank Secrecy Act and OFAC regulations. The firm files Suspicious "
        "Activity Reports (SARs) for transactions exhibiting patterns consistent "
        "with money laundering or terrorist financing. The bank's CET1 capital "
        "ratio stood at 13.1% as of Q4 2023, above the Basel III minimum of 7.0%."
    )

    QUESTION = "What AML obligations does JPMorgan have under the Bank Secrecy Act?"

    bench = ModelBenchmark(run_quality_eval=True)
    report = bench.benchmark(question=QUESTION, context=CONTEXT)

    print(f"\n{'─' * 68}")
    print(f"Fastest      : {report['fastest']}")
    print(f"Cheapest     : {report['cheapest']}")
    print(f"Best quality : {report['best_quality']}")
    print(f"\nRecommendation: {report['recommendation']}")

    print(f"\n{'─' * 68}")
    print("Per-model summary:")
    for row in report["summary"]:
        q = row.get("quality", {})
        quality_str = (
            f"  faith={q.get('faithfulness', 'N/A')}  "
            f"rel={q.get('relevance', 'N/A')}"
            if q else "  quality=skipped"
        )
        print(
            f"  {row['model_name']:<22}  "
            f"latency={row['latency_s']:.2f}s  "
            f"cost=${row['cost_usd']:.5f}"
            f"{quality_str}"
        )
