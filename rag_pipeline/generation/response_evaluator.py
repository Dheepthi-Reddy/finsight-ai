"""
Evaluates the quality of Claude's RAG responses against retrieved context.

What each metric measures
--------------------------
RAG response quality has two distinct dimensions that no single metric captures:

  GROUNDEDNESS  — does the answer draw its claims from the retrieved context,
                  or is the model inventing content?
                  → measured by faithfulness score and ROUGE-L

  LEXICAL COVERAGE — does the answer reproduce the key terminology and concepts
                  from the source, or is it paraphrasing past what the evidence
                  supports?
                  → measured by BLEU and ROUGE-L

Using all three together catches failure modes that each misses individually:
  • High BLEU + low faithfulness  → answer copies phrases from context but its
    *sentences* don't actually make supported claims (cherry-picked words).
  • Low BLEU + high faithfulness  → answer is heavily paraphrased but grounded;
    probably acceptable for compliance Q&A where precise terminology matters less.
  • Low all three                 → answer is likely hallucinated; reject or flag.

Metric-by-metric interpretation
---------------------------------
BLEU (Bilingual Evaluation Understudy)
  Measures n-gram precision of the answer against the context as reference.
  Counts what fraction of 1–4 word sequences in the answer appear verbatim in
  the context, with a brevity penalty for very short answers.
  Originally designed for machine translation quality.

  Score interpretation:
    ≥ 0.40  excellent — answer closely mirrors the language of the source
    0.25–0.39  good — substantial n-gram overlap, mostly grounded
    0.10–0.24  moderate — some overlap; paraphrasing is heavy
    < 0.10  low — answer diverges significantly from context wording

  Caveat: BLEU rewards verbatim copying. A model that rephrases accurately will
  score lower than one that quotes directly. For compliance Q&A this is a useful
  bias — exact regulatory terminology matters — but keep it in context.

ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation — Longest Common Subsequence)
  Measures the longest common subsequence (LCS) between the answer and context.
  Unlike BLEU (which requires contiguous n-grams), ROUGE-L captures non-contiguous
  matches: "the bank filed SARs … in 2022" matches "SARs … bank … 2022" even with
  intervening words. More robust to reordering and paraphrase than BLEU.

  We report the F1 harmonic mean of LCS precision and recall so that both
  coverage (recall) and conciseness (precision) are rewarded.

  Score interpretation:
    ≥ 0.45  strong LCS alignment — answer sequences map closely to context
    0.25–0.44  moderate — partial structural alignment
    < 0.25  weak — answer structure diverges from the source

FAITHFULNESS
  Custom lexical-overlap metric inspired by the RAGAS faithfulness score.
  Splits the answer into sentences, then checks each sentence: does it contain
  a meaningful fraction of non-trivial words that also appear in the context?
  Returns the fraction of answer sentences that pass this overlap threshold.

  A sentence is "faithful" if ≥ 25% of its content words (non-stopwords) appear
  in the context. This is a conservative threshold — even paraphrased sentences
  tend to retain key nouns and verbs that appear in the source.

  Score interpretation:
    ≥ 0.80  high — nearly all sentences are grounded in retrieved passages
    0.60–0.79  medium — most sentences grounded; a few may be inferred
    < 0.60  low — significant portion of answer may be hallucinated; review manually

Usage
-----
    from rag_pipeline.generation.response_evaluator import evaluate_response

    result = evaluate_response(
        answer  = claude_response.text,
        context = retrieved_chunks_joined,
        question= user_question,
    )
    print(result)
    # {
    #   "bleu": 0.312,
    #   "rouge_l": 0.438,
    #   "faithfulness": 0.857,
    #   "overall": "PASS",
    #   "available_metrics": ["bleu", "rouge_l", "faithfulness"],
    # }
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

# ── Optional dependency: nltk ──────────────────────────────────────────────────
# nltk is used for:
#   - word_tokenize(): splits text into individual tokens, handling punctuation
#     better than str.split() (e.g. "AML/BSA" → ["AML", "/", "BSA"])
#   - sent_tokenize(): splits text into sentences, handling abbreviations like
#     "Corp." and "U.S." that would confuse a naive period-split
#   - stopwords corpus: common words (the, a, of) excluded from faithfulness
#     overlap to focus on content-bearing tokens
try:
    import nltk
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False
    warnings.warn(
        "nltk not installed — BLEU score and faithfulness metric will be unavailable.\n"
        "Install with: pip install nltk",
        ImportWarning,
        stacklevel=1,
    )

# ── Optional dependency: rouge-score ──────────────────────────────────────────
# rouge-score is a lightweight library from Google Research implementing
# ROUGE-1, ROUGE-2, and ROUGE-L. It handles tokenisation and LCS computation
# internally.
try:
    from rouge_score import rouge_scorer as _rouge_scorer_module
    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False
    warnings.warn(
        "rouge-score not installed — ROUGE-L metric will be unavailable.\n"
        "Install with: pip install rouge-score",
        ImportWarning,
        stacklevel=1,
    )


# ── NLTK data download (lazy, silent) ─────────────────────────────────────────
# punkt and punkt_tab are sentence/word tokeniser models. stopwords is a list
# of ~150 common English words to exclude from faithfulness overlap.
# download(quiet=True) is a no-op if the data is already present.
def _ensure_nltk_data() -> None:
    if not _NLTK_AVAILABLE:
        return
    for resource, name in [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(name, quiet=True)


# ── Score thresholds ───────────────────────────────────────────────────────────
# These gate the "overall" assessment returned by evaluate_response().
# Deliberately conservative: a compliance assistant should be held to a high
# groundedness bar to avoid hallucinated regulatory claims reaching analysts.
THRESHOLDS = {
    "bleu": {"pass": 0.25, "warn": 0.10},
    "rouge_l": {"pass": 0.30, "warn": 0.15},
    "faithfulness": {"pass": 0.75, "warn": 0.55},
}

# Minimum content-word overlap for a sentence to count as faithful.
# 0.25 means at least 1 in 4 non-stopword answer tokens must appear in context.
FAITHFULNESS_OVERLAP_THRESHOLD = 0.25


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """
    Quality scores for one RAG response.

    A score of None means the library required for that metric was not installed.
    The overall field is derived from whichever metrics are available:
      PASS — all available metrics are above their pass threshold
      WARN — any metric falls between warn and pass thresholds
      FAIL — any metric falls below its warn threshold
      UNKNOWN — no metrics could be computed (all libraries missing)
    """
    bleu: float | None
    rouge_l: float | None
    faithfulness: float | None
    overall: str                      # PASS / WARN / FAIL / UNKNOWN
    available_metrics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bleu": round(self.bleu, 4) if self.bleu is not None else None,
            "rouge_l": round(self.rouge_l, 4) if self.rouge_l is not None else None,
            "faithfulness": round(self.faithfulness, 4) if self.faithfulness is not None else None,
            "overall": self.overall,
            "available_metrics": self.available_metrics,
        }

    def __repr__(self) -> str:
        parts = []
        if self.bleu is not None:
            parts.append(f"bleu={self.bleu:.3f}")
        if self.rouge_l is not None:
            parts.append(f"rouge_l={self.rouge_l:.3f}")
        if self.faithfulness is not None:
            parts.append(f"faithfulness={self.faithfulness:.3f}")
        return f"EvaluationResult({', '.join(parts)}, overall={self.overall!r})"


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_bleu(answer: str, context: str) -> float | None:
    """
    Compute the 4-gram BLEU score with the context as the reference.

    We treat the context as the authoritative reference text and the answer as
    the candidate. A high score means the answer uses language drawn from the
    retrieved passages; a low score means the model paraphrased heavily or
    introduced terminology not present in the source.

    Smoothing method1 adds a fractional count to n-gram precisions that are
    zero, preventing the geometric mean from collapsing to 0.0 when the answer
    contains any n-gram sequence not present in the context (which is common
    for n≥3 in short answers).

    Parameters
    ----------
    answer  : str   Claude's generated response.
    context : str   Concatenated retrieved document chunks.

    Returns
    -------
    float in [0, 1], or None if nltk is not installed.
    """
    if not _NLTK_AVAILABLE:
        return None

    _ensure_nltk_data()

    # word_tokenize lowercases and splits on punctuation cleanly.
    # sentence_bleu expects a list-of-references (each reference is a token list),
    # so we wrap the single context token list in an outer list.
    reference = [nltk.word_tokenize(context.lower())]
    hypothesis = nltk.word_tokenize(answer.lower())

    if not hypothesis:
        return 0.0

    # Equal 4-gram weights: standard BLEU-4 configuration.
    # For very short answers (<4 tokens), higher n-gram orders are always 0;
    # smoothing prevents the overall score from collapsing.
    weights = (0.25, 0.25, 0.25, 0.25)
    smoother = SmoothingFunction().method1

    return float(sentence_bleu(reference, hypothesis, weights=weights,
                               smoothing_function=smoother))


def compute_rouge_l(answer: str, context: str) -> float | None:
    """
    Compute the ROUGE-L F1 score between the answer and context.

    RougeScorer treats answer as the prediction and context as the target.
    The F1 formulation balances:
      precision — how much of the answer's LCS appears in the context
      recall    — how much of the context's LCS is covered by the answer

    For RAG evaluation, we care about both: precision catches hallucinated
    content (answer contains sequences not in context), while recall catches
    omission errors (answer ignores key passages from the context).

    Parameters
    ----------
    answer  : str   Claude's generated response.
    context : str   Concatenated retrieved document chunks.

    Returns
    -------
    float in [0, 1], or None if rouge-score is not installed.
    """
    if not _ROUGE_AVAILABLE:
        return None

    # use_stemmer=True applies Porter stemming so "regulated"/"regulation"/
    # "regulates" all match, capturing morphological variants common in
    # legal/regulatory text.
    scorer = _rouge_scorer_module.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(target=context, prediction=answer)
    return float(scores["rougeL"].fmeasure)


def compute_faithfulness(answer: str, context: str) -> float | None:
    """
    Fraction of answer sentences with sufficient lexical overlap with context.

    Algorithm
    ---------
    1. Tokenise the context into a set of content words (stopwords removed,
       lowercased). This is the "evidence vocabulary".
    2. Sentence-split the answer.
    3. For each sentence, compute:
         overlap_ratio = |{content words in sentence} ∩ evidence_vocab| /
                         |{content words in sentence}|
    4. A sentence is "faithful" if overlap_ratio ≥ FAITHFULNESS_OVERLAP_THRESHOLD.
    5. Return faithful_count / total_sentences.

    Why content words only:
    Stopwords (the, of, and, is) appear in almost every sentence regardless of
    topic. Including them inflates overlap for sentences that share no domain
    content with the context. Restricting to content words makes the metric
    sensitive to actual topical alignment.

    Limitations:
    This is a lexical metric — it cannot detect semantic faithfulness for
    paraphrased content. "The company maintains capital buffers" and "The firm
    holds reserve capital" overlap on only "capital", so the paraphrase would
    score poorly even though it is faithful. For production use, consider
    pairing with an NLI-based faithfulness model (e.g. cross-encoder/nli-deberta).

    Parameters
    ----------
    answer  : str   Claude's generated response.
    context : str   Concatenated retrieved document chunks.

    Returns
    -------
    float in [0, 1], or None if nltk is not installed.
    """
    if not _NLTK_AVAILABLE:
        return None

    _ensure_nltk_data()

    # Load English stopwords. Short words (≤2 chars) are also excluded to
    # filter out residual punctuation tokens and abbreviation fragments.
    try:
        stopwords = set(nltk.corpus.stopwords.words("english"))
    except Exception:
        stopwords = set()

    def _content_tokens(text: str) -> set[str]:
        tokens = nltk.word_tokenize(text.lower())
        return {t for t in tokens if t.isalpha() and t not in stopwords and len(t) > 2}

    evidence_vocab = _content_tokens(context)
    if not evidence_vocab:
        return 0.0

    sentences = nltk.sent_tokenize(answer)
    if not sentences:
        return 0.0

    faithful_count = 0
    for sentence in sentences:
        sentence_tokens = _content_tokens(sentence)
        if not sentence_tokens:
            # Sentences that are purely stopwords/punctuation (e.g., "OK.")
            # are counted as faithful to avoid penalising short filler sentences.
            faithful_count += 1
            continue

        overlap = sentence_tokens & evidence_vocab
        overlap_ratio = len(overlap) / len(sentence_tokens)

        if overlap_ratio >= FAITHFULNESS_OVERLAP_THRESHOLD:
            faithful_count += 1

    return faithful_count / len(sentences)


# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_response(
    answer: str,
    context: str,
    question: str = "",
) -> EvaluationResult:
    """
    Compute BLEU, ROUGE-L, and faithfulness for a single RAG response.

    Parameters
    ----------
    answer   : str
        The text generated by Claude (BedrockClient.invoke().text).
    context  : str
        The concatenated retrieved document chunks that were passed to Claude
        as context. This is the ground truth against which the answer is judged.
    question : str, optional
        The original user question. Currently unused in scoring but included
        in the signature for future answer-relevance metrics (e.g., question-
        answer semantic similarity via cross-encoder).

    Returns
    -------
    EvaluationResult
        Contains individual scores, available_metrics (which libraries were
        found), and an overall PASS/WARN/FAIL/UNKNOWN assessment.

    Notes
    -----
    All three metrics compare the answer to the context, not to a gold-standard
    reference answer. This makes them usable without labelled data, but they
    measure groundedness and lexical fidelity — not factual correctness or
    completeness. A model that copies the context verbatim would score 1.0 on
    all metrics while potentially ignoring the question entirely.

    For production quality gates, combine these scores with:
      - Confidence rating parsed from Claude's COMPLIANCE_SYSTEM_PROMPT output
      - A question-answer relevance check (does the answer address the question?)
      - Manual review sampling for HIGH/CRITICAL compliance decisions
    """
    if not answer.strip() or not context.strip():
        return EvaluationResult(
            bleu=0.0, rouge_l=0.0, faithfulness=0.0,
            overall="FAIL", available_metrics=[],
        )

    available: list[str] = []

    bleu = compute_bleu(answer, context)
    if bleu is not None:
        available.append("bleu")

    rouge_l = compute_rouge_l(answer, context)
    if rouge_l is not None:
        available.append("rouge_l")

    faithfulness = compute_faithfulness(answer, context)
    if faithfulness is not None:
        available.append("faithfulness")

    overall = _compute_overall(
        bleu=bleu,
        rouge_l=rouge_l,
        faithfulness=faithfulness,
    )

    return EvaluationResult(
        bleu=bleu,
        rouge_l=rouge_l,
        faithfulness=faithfulness,
        overall=overall,
        available_metrics=available,
    )


def _compute_overall(
    bleu: float | None,
    rouge_l: float | None,
    faithfulness: float | None,
) -> str:
    """
    Derive a single PASS / WARN / FAIL / UNKNOWN verdict from available scores.

    The overall verdict uses the *worst* individual assessment so that a single
    failing metric triggers a review even when the others look healthy. This
    conservative aggregation is appropriate for compliance use cases where a
    single hallucinated regulatory claim can cause harm.
    """
    scores = {
        "bleu": bleu,
        "rouge_l": rouge_l,
        "faithfulness": faithfulness,
    }

    verdicts: list[str] = []
    for metric, score in scores.items():
        if score is None:
            continue  # library not available — skip
        thresholds = THRESHOLDS[metric]
        if score >= thresholds["pass"]:
            verdicts.append("PASS")
        elif score >= thresholds["warn"]:
            verdicts.append("WARN")
        else:
            verdicts.append("FAIL")

    if not verdicts:
        return "UNKNOWN"

    # Severity order: FAIL > WARN > PASS
    if "FAIL" in verdicts:
        return "FAIL"
    if "WARN" in verdicts:
        return "WARN"
    return "PASS"


# ─────────────────────────────────────────────────────────────────────────────
# BATCH HELPER
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_batch(
    pairs: list[dict[str, str]],
) -> list[EvaluationResult]:
    """
    Evaluate a list of answer/context pairs.

    Parameters
    ----------
    pairs : list[dict]
        Each dict must have keys: "answer", "context".
        Optional key: "question".

    Returns
    -------
    list[EvaluationResult]
        One result per input pair, in the same order.
    """
    results = []
    for pair in pairs:
        results.append(evaluate_response(
            answer=pair["answer"],
            context=pair["context"],
            question=pair.get("question", ""),
        ))
    return results


def summarise_batch(results: list[EvaluationResult]) -> dict[str, Any]:
    """
    Aggregate statistics across a batch of evaluation results.

    Returns mean scores for each available metric, overall verdict counts,
    and the pass rate (fraction of PASS verdicts). Useful for monitoring
    dashboards and regression testing after model or prompt changes.
    """
    if not results:
        return {}

    def _mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    bleu_scores = [r.bleu for r in results if r.bleu is not None]
    rouge_l_scores = [r.rouge_l for r in results if r.rouge_l is not None]
    faithfulness_scores = [r.faithfulness for r in results if r.faithfulness is not None]

    verdicts = [r.overall for r in results]
    return {
        "n": len(results),
        "mean_bleu": _mean(bleu_scores),
        "mean_rouge_l": _mean(rouge_l_scores),
        "mean_faithfulness": _mean(faithfulness_scores),
        "overall_counts": {
            "PASS": verdicts.count("PASS"),
            "WARN": verdicts.count("WARN"),
            "FAIL": verdicts.count("FAIL"),
            "UNKNOWN": verdicts.count("UNKNOWN"),
        },
        "pass_rate": round(verdicts.count("PASS") / len(verdicts), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DEMO / CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CONTEXT = (
        "JPMorgan Chase maintains comprehensive AML compliance programs governed "
        "by the Bank Secrecy Act and OFAC regulations. The firm files Suspicious "
        "Activity Reports (SARs) for transactions exhibiting patterns consistent "
        "with money laundering or terrorist financing. JPMorgan's AML program "
        "includes transaction monitoring, customer due diligence, and enhanced "
        "due diligence for high-risk customers. The bank's CET1 capital ratio "
        "stood at 13.1% as of Q4 2023, well above the regulatory minimum of 4.5%. "
        "Basel III requires banks to hold a minimum CET1 ratio of 4.5% plus a "
        "2.5% capital conservation buffer, for a total of 7.0%."
    )

    # ── Test 1: Faithful, grounded answer ────────────────────────────────────
    grounded_answer = (
        "JPMorgan Chase operates AML compliance programs under the Bank Secrecy Act "
        "and OFAC regulations, filing Suspicious Activity Reports for transactions "
        "suggesting money laundering. The bank maintained a CET1 capital ratio of "
        "13.1% in Q4 2023, comfortably exceeding the 7.0% Basel III requirement."
    )

    # ── Test 2: Partially hallucinated answer ────────────────────────────────
    partial_answer = (
        "JPMorgan files SARs as required by BSA. Their capital levels are strong. "
        "The bank also maintains a Tier 2 capital buffer of 8% under Dodd-Frank "
        "stress testing frameworks, and all trading desks must pass a daily VaR limit."
    )

    # ── Test 3: Completely hallucinated answer ────────────────────────────────
    hallucinated_answer = (
        "The Federal Reserve requires banks to submit quarterly cybersecurity audits "
        "under the GLBA privacy framework. Investment banks must hold a 15% leverage "
        "ratio under the Volcker Rule amendments passed in 2021."
    )

    question = "What AML obligations does JPMorgan have under the Bank Secrecy Act?"

    test_cases = [
        ("Grounded answer", grounded_answer),
        ("Partially hallucinated", partial_answer),
        ("Fully hallucinated", hallucinated_answer),
    ]

    for label, answer in test_cases:
        result = evaluate_response(answer=answer, context=CONTEXT, question=question)
        print(f"\n── {label} ─────────────────────────────────")
        print(f"  {result}")
        print(f"  Scores: {result.to_dict()}")

    # ── Batch summary ─────────────────────────────────────────────────────────
    pairs = [
        {"answer": ans, "context": CONTEXT, "question": question}
        for _, ans in test_cases
    ]
    all_results = evaluate_batch(pairs)
    summary = summarise_batch(all_results)
    print("\n── Batch summary ──────────────────────────────────────────────")
    for k, v in summary.items():
        print(f"  {k}: {v}")
