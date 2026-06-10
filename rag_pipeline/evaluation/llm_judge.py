"""
LLM-as-judge evaluation for the FinSight RAG pipeline.

Why LLM-as-judge is better than lexical metrics
-------------------------------------------------
The lexical metrics in response_evaluator.py (BLEU, ROUGE-L, faithfulness) all
share a fundamental limitation: they measure *word overlap*, not *meaning*.

Consider these two answers to "What are JPMorgan's AML obligations?":

  Answer A: "JPMorgan maintains anti-money laundering compliance programs
             governed by the Bank Secrecy Act, filing Suspicious Activity
             Reports for flagged transactions."

  Answer B: "The firm sustains BSA-mandated controls, submitting SARs whenever
             transaction patterns indicate potential financial crime."

Both answers are faithful, relevant, and complete. But BLEU and ROUGE-L will
score Answer B poorly because it shares almost no n-grams with Answer A or
with the source context — it uses synonyms throughout ("sustains" / "maintains",
"BSA-mandated" / "Bank Secrecy Act", "submitting" / "filing"). BLEU punishes
accurate paraphrase; it has no concept of semantic equivalence.

LLM-as-judge addresses this by using a language model to reason about quality
the same way a human evaluator would:

  FAITHFULNESS  — Can each claim in the answer be traced to a specific passage
                  in the context? A model can identify semantic entailment even
                  when the answer paraphrases rather than quotes.

  RELEVANCE     — Does the answer actually address what the question asks? BLEU
                  cannot tell the difference between an on-topic and off-topic
                  answer if they share vocabulary. An LLM can.

  COMPLETENESS  — Does the answer cover all the important aspects the context
                  supports? A short but accurate answer scores high on BLEU/ROUGE
                  because it avoids wrong n-grams, but it may have omitted three
                  key regulatory requirements. An LLM can identify the gap.

The approach was popularised by the G-Eval framework (Liu et al., 2023) and is
used in production evaluation pipelines at Anthropic, OpenAI, and in the RAGAS
library. The key insight: use the same class of model to judge as to generate,
because the judge needs to understand the domain as well as (or better than)
the generator.

Limitations of LLM-as-judge
-----------------------------
LLM judges are not perfect. Known failure modes:

  VERBOSITY BIAS — LLMs tend to prefer longer, more detailed answers even when
                   a concise answer is factually superior. Mitigated here by
                   explicit scoring rubrics that reward precision.

  SELF-PREFERENCE — A model may rate its own output higher than equivalent
                    output from a different model. For compare(), both answers
                    are presented in the same prompt to reduce this effect.

  POSITION BIAS  — In compare(), models tend to prefer whichever answer is
                   presented first. We return both orderings' raw reasoning
                   so the caller can detect and correct for this.

  NON-DETERMINISM — Scores can vary slightly between calls. Using temperature=0
                    minimises variance but doesn't eliminate it. For high-stakes
                    decisions, run judge() multiple times and average.

How this integrates with response_evaluator.py
----------------------------------------------
LLM-judge and lexical metrics are complementary:
  • Use lexical metrics for fast, cheap, zero-latency evaluation during index
    build and bulk regression testing where you have thousands of examples.
  • Use LLMJudge for sample-based quality audits, A/B comparisons between
    prompt changes, and when semantic correctness matters more than throughput.

Usage
-----
    from rag_pipeline.evaluation.llm_judge import LLMJudge

    judge = LLMJudge()

    result = judge.judge(
        question = "What are JPMorgan's AML obligations under the BSA?",
        answer   = claude_response,
        context  = retrieved_chunks_text,
    )
    print(result)
    # JudgeResult(faithfulness=0.90, relevance=0.95, completeness=0.75,
    #             verdict='PASS', reasoning='...')

    comparison = judge.compare(
        question  = "What are JPMorgan's AML obligations?",
        context   = retrieved_chunks_text,
        answer_a  = response_from_model_v1,
        answer_b  = response_from_model_v2,
    )
    print(comparison["preferred"])  # "A", "B", or "TIE"
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from rag_pipeline.generation.bedrock_client import BedrockClient

# ── Configuration ─────────────────────────────────────────────────────────────

# Score threshold for a PASS verdict. All three dimensions must meet or exceed
# this value. 0.7 is deliberately conservative for a compliance domain where a
# partially-grounded answer could mislead a regulatory decision.
PASS_THRESHOLD = 0.7

# Score threshold below which any dimension triggers a FAIL verdict regardless
# of the other two. A faithfulness of 0.4 means more than half the answer
# sentences are potentially hallucinated — reject regardless of relevance.
FAIL_THRESHOLD = 0.4

# Max tokens for the judge's reasoning response. The judge must return a JSON
# object plus a brief explanation — 512 tokens is ample without padding cost.
JUDGE_MAX_TOKENS = 512


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class JudgeResult:
    """
    Scores and verdict from a single LLMJudge.judge() call.

    Fields
    ------
    faithfulness  : float   [0, 1]  Fraction of answer grounded in context.
                                    1.0 = every claim traceable to the context;
                                    0.0 = no claims supported by context.
    relevance     : float   [0, 1]  How directly the answer addresses the
                                    question. 1.0 = fully on-topic and complete
                                    for the question asked; 0.0 = off-topic.
    completeness  : float   [0, 1]  How much of the relevant context information
                                    the answer captures. 1.0 = covers all key
                                    points; 0.0 = misses most important content.
    verdict       : str             PASS / WARN / FAIL based on score thresholds.
    reasoning     : str             The judge's brief explanation of each score.
                                    Useful for debugging and audit trails.
    is_stub       : bool            True when BedrockClient ran in stub mode —
                                    scores are synthetic and not meaningful.
    raw_response  : str             The judge's complete raw text output, before
                                    JSON parsing. Preserved for debugging cases
                                    where JSON extraction fails.
    """
    faithfulness:  float
    relevance:     float
    completeness:  float
    verdict:       str
    reasoning:     str       = ""
    is_stub:       bool      = False
    raw_response:  str       = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "faithfulness":  round(self.faithfulness, 4),
            "relevance":     round(self.relevance, 4),
            "completeness":  round(self.completeness, 4),
            "verdict":       self.verdict,
            "reasoning":     self.reasoning,
            "is_stub":       self.is_stub,
        }

    def __repr__(self) -> str:
        return (
            f"JudgeResult("
            f"faithfulness={self.faithfulness:.2f}, "
            f"relevance={self.relevance:.2f}, "
            f"completeness={self.completeness:.2f}, "
            f"verdict={self.verdict!r}, "
            f"stub={self.is_stub})"
        )


@dataclass
class CompareResult:
    """
    Output of LLMJudge.compare() — head-to-head evaluation of two answers.

    Fields
    ------
    score_a     : JudgeResult   Scores for answer_a evaluated independently.
    score_b     : JudgeResult   Scores for answer_b evaluated independently.
    preferred   : str           "A", "B", or "TIE" — which answer the judge
                                considers superior overall.
    reasoning   : str           The judge's comparative reasoning explaining
                                why one answer was preferred.
    margin      : float         Absolute difference in average scores between
                                A and B. Values < 0.05 typically indicate a
                                practical tie even if preferred != "TIE".
    is_stub     : bool          True when both evaluations used stub mode.
    """
    score_a:    JudgeResult
    score_b:    JudgeResult
    preferred:  str               # "A", "B", or "TIE"
    reasoning:  str       = ""
    margin:     float     = 0.0
    is_stub:    bool      = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "score_a":   self.score_a.to_dict(),
            "score_b":   self.score_b.to_dict(),
            "preferred": self.preferred,
            "margin":    round(self.margin, 4),
            "reasoning": self.reasoning,
            "is_stub":   self.is_stub,
        }

    def __repr__(self) -> str:
        return (
            f"CompareResult("
            f"preferred={self.preferred!r}, "
            f"margin={self.margin:.3f}, "
            f"A={self.score_a.verdict}, "
            f"B={self.score_b.verdict})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# LLM JUDGE
# ─────────────────────────────────────────────────────────────────────────────

class LLMJudge:
    """
    Uses Claude via AWS Bedrock to evaluate RAG answers on three dimensions.

    Instantiate once and reuse — BedrockClient resolves credentials at init
    time and the underlying boto3 session is shared across all calls.

    The judge prompt is designed to elicit structured JSON output so scores
    can be parsed programmatically. A fallback regex extractor handles cases
    where the model wraps the JSON in prose or markdown code fences.
    """

    def __init__(
        self,
        bedrock:        BedrockClient | None = None,
        pass_threshold: float = PASS_THRESHOLD,
        fail_threshold: float = FAIL_THRESHOLD,
    ) -> None:
        """
        Parameters
        ----------
        bedrock        : BedrockClient, optional
            Pre-built client. If None, a new one is created (auto-detects
            credentials and falls back to stub mode if unavailable).
        pass_threshold : float
            Minimum score on ALL three dimensions for a PASS verdict.
        fail_threshold : float
            Maximum score on ANY dimension before triggering a FAIL verdict.
        """
        self.bedrock        = bedrock if bedrock is not None else BedrockClient()
        self.pass_threshold = pass_threshold
        self.fail_threshold = fail_threshold

    # ── Primary evaluation ────────────────────────────────────────────────────

    def judge(
        self,
        question: str,
        answer:   str,
        context:  str,
    ) -> JudgeResult:
        """
        Evaluate a single RAG answer on faithfulness, relevance, and completeness.

        The judge receives the same three inputs as the generator (question,
        context, answer) and scores the answer along three orthogonal dimensions.
        This mirrors how a human expert would audit a compliance response: check
        that claims are sourced, the question is actually answered, and nothing
        important was omitted.

        Parameters
        ----------
        question : str   The original user question.
        answer   : str   The RAG-generated answer to evaluate.
        context  : str   The retrieved document chunks passed to the generator.
                         The judge uses this as the ground truth for faithfulness.

        Returns
        -------
        JudgeResult with scores, verdict, and reasoning.
        """
        if not question.strip() or not answer.strip() or not context.strip():
            return JudgeResult(
                faithfulness=0.0,
                relevance=0.0,
                completeness=0.0,
                verdict="FAIL",
                reasoning="One or more inputs (question, answer, context) were empty.",
            )

        prompt = _build_judge_prompt(question=question, answer=answer, context=context)

        response = self.bedrock.invoke(
            prompt        = prompt,
            system_prompt = _JUDGE_SYSTEM_PROMPT,
            max_tokens    = JUDGE_MAX_TOKENS,
            temperature   = 0.0,  # deterministic scoring — reduce noise across runs
        )

        scores = _parse_scores(response.text)

        faithfulness  = scores.get("faithfulness",  0.0)
        relevance     = scores.get("relevance",     0.0)
        completeness  = scores.get("completeness",  0.0)
        reasoning     = scores.get("reasoning",     "")
        verdict       = _compute_verdict(faithfulness, relevance, completeness,
                                         self.pass_threshold, self.fail_threshold)

        return JudgeResult(
            faithfulness  = faithfulness,
            relevance     = relevance,
            completeness  = completeness,
            verdict       = verdict,
            reasoning     = reasoning,
            is_stub       = response.is_stub,
            raw_response  = response.text,
        )

    # ── Comparative evaluation ────────────────────────────────────────────────

    def compare(
        self,
        question:  str,
        context:   str,
        answer_a:  str,
        answer_b:  str,
    ) -> CompareResult:
        """
        Compare two answers to the same question and declare a winner.

        Both answers are evaluated independently first (to get individual scores),
        then a separate comparative prompt asks the judge to reason holistically
        about which is superior. Using two-step evaluation avoids conflating the
        per-dimension scores with the overall preference decision.

        Position bias note: LLMs tend to favour whichever answer appears first in
        the prompt. To mitigate this, the comparative prompt explicitly instructs
        the judge to evaluate both answers before forming a preference, and the
        margin field signals when the difference is too small to be meaningful.

        Parameters
        ----------
        question : str   The user question both answers address.
        context  : str   The retrieved context both answers were generated from.
        answer_a : str   First candidate answer (e.g. from model version A).
        answer_b : str   Second candidate answer (e.g. from model version B).

        Returns
        -------
        CompareResult with individual scores, preferred winner, and reasoning.
        """
        # ── Step 1: evaluate each answer independently ───────────────────────
        # Independent evaluation avoids contamination: if B is shown alongside A,
        # the judge may inflate B's score relative to A's instead of grading each
        # against the absolute rubric.
        score_a = self.judge(question=question, answer=answer_a, context=context)
        score_b = self.judge(question=question, answer=answer_b, context=context)

        # ── Step 2: comparative head-to-head prompt ───────────────────────────
        compare_prompt = _build_compare_prompt(
            question=question,
            context=context,
            answer_a=answer_a,
            answer_b=answer_b,
        )

        response = self.bedrock.invoke(
            prompt        = compare_prompt,
            system_prompt = _JUDGE_SYSTEM_PROMPT,
            max_tokens    = JUDGE_MAX_TOKENS,
            temperature   = 0.0,
        )

        comparison = _parse_comparison(response.text)
        preferred  = comparison.get("preferred", "TIE")
        reasoning  = comparison.get("reasoning", "")

        # Compute the average score for each answer; the margin is the absolute
        # difference. A margin < 0.05 is practically indistinguishable regardless
        # of the preferred label.
        avg_a  = (score_a.faithfulness + score_a.relevance + score_a.completeness) / 3
        avg_b  = (score_b.faithfulness + score_b.relevance + score_b.completeness) / 3
        margin = round(abs(avg_a - avg_b), 4)

        # Override stub-generated "preferred" with a score-based decision when
        # in stub mode — the stub returns canned text that may not parse cleanly.
        if score_a.is_stub:
            preferred = "TIE" if margin < 0.05 else ("A" if avg_a > avg_b else "B")

        return CompareResult(
            score_a   = score_a,
            score_b   = score_b,
            preferred = preferred,
            reasoning = reasoning,
            margin    = margin,
            is_stub   = score_a.is_stub,
        )

    # ── Batch evaluation ──────────────────────────────────────────────────────

    def judge_batch(
        self,
        examples: list[dict[str, str]],
    ) -> list[JudgeResult]:
        """
        Evaluate a list of question/answer/context triples.

        Parameters
        ----------
        examples : list[dict]
            Each dict must have keys: "question", "answer", "context".

        Returns
        -------
        list[JudgeResult]  One result per example, in the same order.
        """
        results = []
        for i, ex in enumerate(examples):
            print(f"  Judging example {i + 1}/{len(examples)}...")
            results.append(self.judge(
                question = ex["question"],
                answer   = ex["answer"],
                context  = ex["context"],
            ))
        return results

    def summarise_batch(self, results: list[JudgeResult]) -> dict[str, Any]:
        """
        Aggregate statistics across a batch of JudgeResults.

        Returns mean scores per dimension, verdict counts, and pass rate.
        Useful for regression dashboards: run judge_batch() on a fixed eval
        set before and after a prompt change to detect quality regressions.
        """
        if not results:
            return {}

        def _mean(values: list[float]) -> float:
            return round(sum(values) / len(values), 4) if values else 0.0

        faithfulness_scores = [r.faithfulness for r in results]
        relevance_scores    = [r.relevance    for r in results]
        completeness_scores = [r.completeness for r in results]
        verdicts            = [r.verdict      for r in results]

        avg_overall = _mean([
            (r.faithfulness + r.relevance + r.completeness) / 3
            for r in results
        ])

        return {
            "n":                   len(results),
            "mean_faithfulness":   _mean(faithfulness_scores),
            "mean_relevance":      _mean(relevance_scores),
            "mean_completeness":   _mean(completeness_scores),
            "mean_overall":        avg_overall,
            "verdict_counts": {
                "PASS": verdicts.count("PASS"),
                "WARN": verdicts.count("WARN"),
                "FAIL": verdicts.count("FAIL"),
            },
            "pass_rate": round(verdicts.count("PASS") / len(verdicts), 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

# The judge system prompt establishes the evaluator persona and output contract.
# Key design decisions:
#   - "Impartial evaluator" framing reduces verbosity bias: the judge is not
#     trying to be helpful to the answer's author, it is auditing accuracy.
#   - JSON-only output is mandated so parsing is deterministic; the judge is
#     explicitly told not to add prose outside the JSON object.
#   - Score definitions are given as explicit criteria, not vague adjectives.
#     "0.9 means one minor unsupported claim" is more consistent than "0.9 means
#     mostly grounded" because "mostly" is calibrated differently per model call.
_JUDGE_SYSTEM_PROMPT = """\
You are an impartial evaluation judge for a financial compliance RAG system.
Your task is to assess the quality of AI-generated answers to compliance questions.

You will be given:
  - QUESTION: the user's compliance question
  - CONTEXT: the retrieved document passages the answer was generated from
  - ANSWER: the AI-generated answer to evaluate

Score the ANSWER on three dimensions. Return ONLY a valid JSON object — no prose
before or after it, no markdown fences.

JSON format:
{
  "faithfulness": <float 0.0–1.0>,
  "relevance": <float 0.0–1.0>,
  "completeness": <float 0.0–1.0>,
  "reasoning": "<one concise sentence per dimension explaining each score>"
}

Scoring rubric:

FAITHFULNESS — Is every factual claim in the answer supported by the context?
  1.0 : All claims directly traceable to the provided passages; no unsupported assertions.
  0.8 : One minor claim inferred beyond the context but plausible given the evidence.
  0.6 : Several claims go beyond the context; answer mixes retrieved facts with inference.
  0.4 : Major claims have no basis in the context; likely hallucinated content present.
  0.0 : Answer contradicts or ignores the context entirely.

RELEVANCE — Does the answer directly address what the question asked?
  1.0 : Directly and completely answers the specific question asked.
  0.8 : Answers the question but includes minor tangential content.
  0.6 : Partially answers the question; misses a key aspect of what was asked.
  0.4 : Addresses a related but different question; the actual question is not answered.
  0.0 : Completely off-topic; does not address the question at all.

COMPLETENESS — How much of the relevant information from the context is captured?
  1.0 : All key points from the context relevant to the question are included.
  0.8 : Most key points covered; one minor relevant detail omitted.
  0.6 : Covers the main point but omits supporting details or qualifications.
  0.4 : Only partially covers the relevant content; significant omissions.
  0.0 : Captures almost none of the relevant information from the context.
"""


def _build_judge_prompt(question: str, answer: str, context: str) -> str:
    """
    Render the per-call evaluation prompt with the three inputs substituted in.

    The context is truncated to 3,000 characters to stay within token budgets —
    the judge needs enough context to verify faithfulness but doesn't need the
    full corpus. The question and answer are not truncated.
    """
    context_preview = context[:3000] + ("..." if len(context) > 3000 else "")
    return (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context_preview}\n\n"
        f"ANSWER:\n{answer}"
    )


def _build_compare_prompt(
    question:  str,
    context:   str,
    answer_a:  str,
    answer_b:  str,
) -> str:
    """
    Render the head-to-head comparison prompt.

    Instructs the judge to evaluate both answers *before* picking a winner to
    reduce the primacy bias effect (tendency to prefer whatever is read first).
    """
    context_preview = context[:2000] + ("..." if len(context) > 2000 else "")
    return (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context_preview}\n\n"
        f"ANSWER A:\n{answer_a}\n\n"
        f"ANSWER B:\n{answer_b}\n\n"
        "Evaluate both answers carefully against the question and context before "
        "deciding. Return ONLY a valid JSON object:\n"
        '{"preferred": "A" | "B" | "TIE", "reasoning": "<explanation>"}'
    )


# ─────────────────────────────────────────────────────────────────────────────
# PARSING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_scores(text: str) -> dict[str, Any]:
    """
    Extract the JSON score object from the judge's response text.

    The judge is instructed to return only JSON, but in practice models
    sometimes wrap it in markdown fences or prepend a sentence. This function
    tries direct JSON parsing first, then falls back to regex extraction, then
    returns safe zero-scores so the rest of the pipeline doesn't break.

    Parameters
    ----------
    text : str   Raw response text from BedrockClient.invoke().

    Returns
    -------
    dict with keys: faithfulness, relevance, completeness, reasoning.
    """
    # Strategy 1: direct parse — works when the model obeys the instruction.
    try:
        data = json.loads(text.strip())
        return _validate_scores(data)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: extract JSON object from surrounding prose or code fences.
    # Matches the outermost {...} block in the response.
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return _validate_scores(data)
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: individual float extraction — last resort when JSON is malformed.
    # Scans for patterns like '"faithfulness": 0.85' anywhere in the text.
    scores: dict[str, Any] = {}
    for key in ("faithfulness", "relevance", "completeness"):
        pattern = rf'"{key}"\s*:\s*([0-9]*\.?[0-9]+)'
        m = re.search(pattern, text)
        if m:
            scores[key] = _clamp(float(m.group(1)))

    if scores:
        scores.setdefault("faithfulness",  0.0)
        scores.setdefault("relevance",     0.0)
        scores.setdefault("completeness",  0.0)
        scores.setdefault("reasoning",     "Score extracted via fallback regex.")
        return scores

    # All strategies failed — return zeros so the pipeline continues and the
    # raw_response field can be inspected to diagnose the parse failure.
    return {
        "faithfulness":  0.0,
        "relevance":     0.0,
        "completeness":  0.0,
        "reasoning":     f"Failed to parse judge response: {text[:200]}",
    }


def _parse_comparison(text: str) -> dict[str, str]:
    """
    Extract the preferred winner and reasoning from the comparative prompt response.

    Returns a dict with keys: "preferred" ("A", "B", or "TIE") and "reasoning".
    """
    try:
        data = json.loads(text.strip())
        preferred = str(data.get("preferred", "TIE")).upper()
        if preferred not in ("A", "B", "TIE"):
            preferred = "TIE"
        return {"preferred": preferred, "reasoning": str(data.get("reasoning", ""))}
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: look for the literal strings "A", "B", "TIE" in the response.
    match = re.search(r'"preferred"\s*:\s*"([ABab]|[Tt][Ii][Ee])"', text)
    if match:
        return {
            "preferred": match.group(1).upper(),
            "reasoning": "Preference extracted via fallback regex.",
        }

    return {"preferred": "TIE", "reasoning": "Could not parse comparison result."}


def _validate_scores(data: dict) -> dict[str, Any]:
    """
    Validate and clamp score values to [0, 1].

    Raises ValueError if the required score keys are missing so the caller can
    fall back to regex extraction.
    """
    for key in ("faithfulness", "relevance", "completeness"):
        if key not in data:
            raise ValueError(f"Missing required key: {key!r}")
        data[key] = _clamp(float(data[key]))
    data.setdefault("reasoning", "")
    return data


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a float to [lo, hi]. Guards against models returning e.g. 1.2."""
    return max(lo, min(hi, value))


# ─────────────────────────────────────────────────────────────────────────────
# VERDICT LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def _compute_verdict(
    faithfulness:   float,
    relevance:      float,
    completeness:   float,
    pass_threshold: float,
    fail_threshold: float,
) -> str:
    """
    Derive a PASS / WARN / FAIL verdict from the three dimension scores.

    Logic (applied in severity order):
      FAIL — any single dimension is below fail_threshold. One critically bad
             dimension disqualifies the answer regardless of the others.
             A faithfulness of 0.3 means hallucination is present; the answer
             should not be served to a user regardless of its relevance.
      PASS — all three dimensions meet or exceed pass_threshold.
      WARN — none triggered FAIL but at least one is below pass_threshold.
             Answer can be shown but should be flagged for review.
    """
    scores = [faithfulness, relevance, completeness]

    if any(s < fail_threshold for s in scores):
        return "FAIL"
    if all(s >= pass_threshold for s in scores):
        return "PASS"
    return "WARN"


# ─────────────────────────────────────────────────────────────────────────────
# DEMO / CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CONTEXT = (
        "JPMorgan Chase maintains comprehensive AML compliance programs governed "
        "by the Bank Secrecy Act (BSA) and OFAC regulations. The firm files "
        "Suspicious Activity Reports (SARs) for transactions exhibiting patterns "
        "consistent with money laundering or terrorist financing. JPMorgan's AML "
        "program includes transaction monitoring, customer due diligence (CDD), "
        "and enhanced due diligence (EDD) for high-risk customers. "
        "The bank's CET1 capital ratio stood at 13.1% as of Q4 2023, well above "
        "the regulatory minimum of 4.5%. Basel III requires banks to hold a "
        "minimum CET1 ratio of 4.5% plus a 2.5% capital conservation buffer."
    )

    QUESTION = "What AML obligations does JPMorgan have under the Bank Secrecy Act?"

    # ── Test answers with varying quality ────────────────────────────────────
    GROUNDED = (
        "JPMorgan Chase operates AML compliance programs under the Bank Secrecy "
        "Act and OFAC regulations, filing Suspicious Activity Reports for "
        "transactions suggesting money laundering. The bank applies CDD and EDD "
        "for high-risk customers and runs transaction monitoring systems."
    )

    PARTIAL = (
        "JPMorgan files SARs as required by law. The bank also maintains strong "
        "capital levels with a CET1 ratio that exceeds regulatory requirements, "
        "and has robust internal controls across all business lines."
    )

    HALLUCINATED = (
        "The Federal Reserve requires JPMorgan to submit quarterly CCAR stress "
        "test results under Dodd-Frank. The bank must maintain a leverage ratio "
        "of at least 8% under the Volcker Rule amendments passed in 2021."
    )

    judge = LLMJudge()
    print(f"LLMJudge ready  stub={judge.bedrock._stub_mode}\n")

    # ── Single-answer evaluation ──────────────────────────────────────────────
    for label, answer in [
        ("Grounded answer",        GROUNDED),
        ("Partially off-topic",    PARTIAL),
        ("Hallucinated answer",    HALLUCINATED),
    ]:
        result = judge.judge(question=QUESTION, answer=answer, context=CONTEXT)
        print(f"── {label}")
        print(f"   {result}")
        print(f"   reasoning: {result.reasoning[:120]}")
        print()

    # ── Head-to-head comparison ───────────────────────────────────────────────
    print("── Comparison: grounded vs partially off-topic")
    comparison = judge.compare(
        question  = QUESTION,
        context   = CONTEXT,
        answer_a  = GROUNDED,
        answer_b  = PARTIAL,
    )
    print(f"   {comparison}")
    print(f"   preferred={comparison.preferred}  margin={comparison.margin:.3f}")
    print(f"   reasoning: {comparison.reasoning[:120]}")
    print()

    # ── Batch evaluation ──────────────────────────────────────────────────────
    print("── Batch evaluation (3 examples)")
    examples = [
        {"question": QUESTION, "answer": GROUNDED,     "context": CONTEXT},
        {"question": QUESTION, "answer": PARTIAL,      "context": CONTEXT},
        {"question": QUESTION, "answer": HALLUCINATED, "context": CONTEXT},
    ]
    results = judge.judge_batch(examples)
    summary = judge.summarise_batch(results)
    print(f"   summary: {summary}")
