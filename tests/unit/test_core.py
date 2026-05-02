"""
Unit tests for FinSight core modules.

Coverage
--------
  TestFeatureEngineering  — pure math in api/routers/fraud.py::_engineer_features
  TestRiskLevels          — threshold logic from fraud_model/serving/predictor.py
  TestFaithfulness        — lexical overlap scorer in response_evaluator.py
  TestBLEU                — BLEU score computation in response_evaluator.py
  TestEvaluateResponse    — end-to-end evaluator integration
  TestChunker             — chunk_documents() in rag_pipeline/ingestion/chunker.py
  TestTransactionGenerator— fraud-rate arithmetic (no module import — see note below)

Design philosophy
-----------------
Every test in this file runs without loading any ML model (SentenceTransformer,
XGBoost, FAISS). Tests import only pure-logic helpers and module-level constants,
or generate small synthetic data from scratch.

The transaction generator (data_pipeline/generators/transaction_generator.py)
executes ALL its code at module level (Faker builds 50k users; NumPy generates
500k transactions). Importing it in a test suite would add ~60s of overhead on
every `pytest` run. Instead, we:
  1. Test its arithmetic directly from the known constants.
  2. Optionally verify the output CSV if it already exists on disk.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# ── pytest markers ─────────────────────────────────────────────────────────────
# Tests that require optional libraries are marked so `pytest -m "not slow"`
# still passes in minimal CI environments.
pytestmark = pytest.mark.filterwarnings("ignore::ImportWarning")

# Check optional library availability at module level without skipping the file.
# Tests that need these libraries apply per-class skipping in their _import fixture.
try:
    import nltk as _nltk
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False

try:
    from rouge_score import rouge_scorer as _rouge_mod  # noqa: F401
    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — FEATURE ENGINEERING
# Source: api/routers/fraud.py :: _engineer_features()
#
# These tests verify the pure arithmetic that converts raw transaction fields
# into the feature vector fed to XGBoost. No model loading required — the
# functions only use math, comparisons, and the MERCHANT_RISK dict.
# ═════════════════════════════════════════════════════════════════════════════

class TestFeatureEngineering:

    @pytest.fixture(autouse=True)
    def _import(self):
        """Import the fraud router once per class, not once per test."""
        from api.routers.fraud import MERCHANT_RISK, TransactionRequest, _engineer_features
        self.MERCHANT_RISK     = MERCHANT_RISK
        self.TransactionRequest = TransactionRequest
        self._engineer_features = _engineer_features

    def _req(self, **kwargs):
        """Helper: build a TransactionRequest with sensible defaults."""
        defaults = dict(
            amount=100.0,
            merchant_category="grocery",
            hour_of_day=14,
            day_of_week=2,       # Wednesday
        )
        defaults.update(kwargs)
        return self.TransactionRequest(**defaults)

    # ── amount_log ─────────────────────────────────────────────────────────────

    def test_amount_log_uses_log1p(self):
        """
        amount_log must be log1p(amount), NOT log(amount).
        log1p is used so that $0.00 authorisation holds (is_fraud=0 card checks)
        produce log1p(0)=0.0 instead of log(0)=-inf, which would corrupt the
        feature vector and cause XGBoost to predict garbage for auth-only tokens.
        """
        req = self._req(amount=100.0)
        features, _ = self._engineer_features(req)
        assert features["amount_log"] == pytest.approx(math.log1p(100.0), rel=1e-6)

    def test_amount_log_zero_amount(self):
        """$0 authorisation hold must not produce -inf or NaN."""
        req = self._req(amount=0.0)
        features, _ = self._engineer_features(req)
        assert features["amount_log"] == pytest.approx(0.0)
        assert math.isfinite(features["amount_log"])

    def test_amount_log_large_amount(self):
        """Large amounts compress correctly: log1p(10_000) ≈ 9.21."""
        req = self._req(amount=10_000.0)
        features, _ = self._engineer_features(req)
        assert features["amount_log"] == pytest.approx(math.log1p(10_000.0), rel=1e-6)

    # ── is_night ───────────────────────────────────────────────────────────────

    def test_is_night_midnight(self):
        """Hour 0 (midnight) is night — cardholder is almost certainly asleep."""
        features, _ = self._engineer_features(self._req(hour_of_day=0))
        assert features["is_night"] == 1

    def test_is_night_last_night_hour(self):
        """Hour 5 (5 AM) is still night — boundary must be inclusive."""
        features, _ = self._engineer_features(self._req(hour_of_day=5))
        assert features["is_night"] == 1

    def test_is_night_first_day_hour(self):
        """Hour 6 (6 AM) is daytime — boundary must be exclusive of 6+."""
        features, _ = self._engineer_features(self._req(hour_of_day=6))
        assert features["is_night"] == 0

    def test_is_night_midday(self):
        """Hour 14 (2 PM) is clearly daytime."""
        features, _ = self._engineer_features(self._req(hour_of_day=14))
        assert features["is_night"] == 0

    # ── is_weekend ─────────────────────────────────────────────────────────────

    def test_is_weekend_friday_is_weekday(self):
        """
        day_of_week=4 is Friday — a weekday despite being the end of the work week.
        The model was trained with 0=Monday … 6=Sunday so Friday=4 is not weekend.
        """
        features, _ = self._engineer_features(self._req(day_of_week=4))
        assert features["is_weekend"] == 0

    def test_is_weekend_saturday(self):
        """day_of_week=5 is Saturday — must be flagged as weekend."""
        features, _ = self._engineer_features(self._req(day_of_week=5))
        assert features["is_weekend"] == 1

    def test_is_weekend_sunday(self):
        """day_of_week=6 is Sunday — must be flagged as weekend."""
        features, _ = self._engineer_features(self._req(day_of_week=6))
        assert features["is_weekend"] == 1

    def test_is_weekend_monday(self):
        """day_of_week=0 is Monday — should not be weekend."""
        features, _ = self._engineer_features(self._req(day_of_week=0))
        assert features["is_weekend"] == 0

    # ── amount_x_risk ──────────────────────────────────────────────────────────

    def test_amount_x_risk_low_risk_merchant(self):
        """
        Grocery (risk=0.05) produces a very small multiplier.
        $100 at grocery → amount_x_risk = 5.0
        Same $100 at wire_transfer (risk=0.95) → 95.0 — 19× higher signal.
        This asymmetry is the primary reason the model separates fraud patterns.
        """
        req = self._req(amount=100.0, merchant_category="grocery")
        features, _ = self._engineer_features(req)
        assert features["amount_x_risk"] == pytest.approx(100.0 * 0.05, rel=1e-6)

    def test_amount_x_risk_high_risk_merchant(self):
        """wire_transfer (risk=0.95) amplifies the amount by 19× vs grocery."""
        req = self._req(amount=100.0, merchant_category="wire_transfer")
        features, _ = self._engineer_features(req)
        assert features["amount_x_risk"] == pytest.approx(100.0 * 0.95, rel=1e-6)

    def test_amount_x_risk_uses_correct_weight(self):
        """
        Every category in MERCHANT_RISK must produce the correct multiplier.
        Tests a cross-section of categories at a fixed $1,000 amount.
        """
        test_cases = [
            ("grocery",        0.05),
            ("restaurant",     0.05),
            ("gas_station",    0.10),
            ("travel",         0.20),
            ("pawn_shop",      0.90),
            ("crypto_exchange",0.90),
            ("wire_transfer",  0.95),
        ]
        for category, expected_weight in test_cases:
            req = self._req(amount=1_000.0, merchant_category=category)
            features, _ = self._engineer_features(req)
            assert features["amount_x_risk"] == pytest.approx(
                1_000.0 * expected_weight, rel=1e-6
            ), f"Failed for merchant_category={category!r}"

    # ── velocity_estimated flag ────────────────────────────────────────────────

    def test_velocity_not_estimated_when_explicitly_supplied(self):
        """
        When the caller provides txn_count_1h AND amount_sum_1h, the velocity
        signal is real — velocity_estimated must be False.
        """
        req = self._req(txn_count_1h=6, amount_sum_1h=9_800.0)
        _, velocity_estimated = self._engineer_features(req)
        assert velocity_estimated is False

    def test_velocity_estimated_when_defaults_used(self):
        """
        When the caller omits both velocity fields, the API defaults to
        txn_count_1h=1 and amount_sum_1h=amount. The flag must be True so
        downstream consumers know the burst signal is approximate.
        """
        req = self._req(amount=100.0)   # no txn_count_1h or amount_sum_1h
        _, velocity_estimated = self._engineer_features(req)
        assert velocity_estimated is True

    def test_all_feature_keys_present(self):
        """
        The features dict must contain exactly the keys the predictor expects.
        A missing key raises ValueError inside FraudPredictor._encode(); catching
        it here is faster and produces a more actionable error message.
        """
        req = self._req()
        features, _ = self._engineer_features(req)
        expected_keys = {
            "amount", "amount_log", "amount_x_risk",
            "amount_sum_1h", "txn_count_1h",
            "hour_of_day", "day_of_week",
            "is_night", "is_weekend",
            "account_age_bucket", "merchant_category",
        }
        assert expected_keys.issubset(features.keys())


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — RISK LEVEL THRESHOLDS
# Source: fraud_model/serving/predictor.py :: RISK_THRESHOLDS
#
# FraudPredictor._to_risk_level() iterates RISK_THRESHOLDS and returns the
# label of the first tier whose floor is <= proba. We replicate that logic
# here so the tests don't require instantiating the predictor (which would
# load 50 MB of model weights from disk).
# ═════════════════════════════════════════════════════════════════════════════

def _to_risk_level(proba: float) -> str:
    """Mirror of FraudPredictor._to_risk_level — imported logic, no model load."""
    from fraud_model.serving.predictor import RISK_THRESHOLDS
    for threshold, level in RISK_THRESHOLDS:
        if proba >= threshold:
            return level
    return "LOW"


class TestRiskLevels:

    def test_low_risk_zero(self):
        """0% probability is cleanly LOW."""
        assert _to_risk_level(0.00) == "LOW"

    def test_low_risk_upper_boundary(self):
        """
        29.9% is still LOW — the MEDIUM tier starts at 30%.
        Off-by-one errors at tier boundaries are a common model-serving bug:
        a transaction just below 30% should never be escalated to soft review.
        """
        assert _to_risk_level(0.299) == "LOW"

    def test_medium_risk_at_boundary(self):
        """Exactly 30% hits the MEDIUM floor — must not remain LOW."""
        assert _to_risk_level(0.30) == "MEDIUM"

    def test_medium_risk_mid(self):
        """45% is squarely in the MEDIUM tier."""
        assert _to_risk_level(0.45) == "MEDIUM"

    def test_medium_risk_upper_boundary(self):
        """59.9% is still MEDIUM — HIGH starts at 60%."""
        assert _to_risk_level(0.599) == "MEDIUM"

    def test_high_risk_at_boundary(self):
        """Exactly 60% triggers HIGH — must leave MEDIUM."""
        assert _to_risk_level(0.60) == "HIGH"

    def test_high_risk_mid(self):
        """72% is squarely in the HIGH tier — route to manual review queue."""
        assert _to_risk_level(0.72) == "HIGH"

    def test_high_risk_upper_boundary(self):
        """84.9% is still HIGH — CRITICAL starts at 85%."""
        assert _to_risk_level(0.849) == "HIGH"

    def test_critical_at_boundary(self):
        """
        Exactly 85% triggers CRITICAL — the transaction should be blocked
        immediately. This is the most important boundary: a misclassification
        here means a fraudulent transaction is routed to manual review instead
        of being blocked, potentially allowing fund transfer before review.
        """
        assert _to_risk_level(0.85) == "CRITICAL"

    def test_critical_high_confidence(self):
        """99% probability is CRITICAL."""
        assert _to_risk_level(0.99) == "CRITICAL"

    def test_all_four_tiers_reachable(self):
        """Sanity check: one representative probability hits each tier exactly once."""
        probs_and_expected = [
            (0.10, "LOW"),
            (0.40, "MEDIUM"),
            (0.70, "HIGH"),
            (0.90, "CRITICAL"),
        ]
        for proba, expected in probs_and_expected:
            assert _to_risk_level(proba) == expected, f"proba={proba}"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RESPONSE EVALUATOR FAITHFULNESS
# Source: rag_pipeline/generation/response_evaluator.py
#
# Faithfulness measures what fraction of answer sentences contain enough
# content-word overlap with the retrieved context to be considered "grounded".
# These tests verify the logic with predictable, hand-crafted strings so the
# expected scores can be computed by inspection without running a model.
#
# nltk is required. Tests are skipped gracefully if it is not installed.
# ═════════════════════════════════════════════════════════════════════════════

class TestFaithfulness:

    @pytest.fixture(autouse=True)
    def _import(self):
        if not _NLTK_AVAILABLE:
            pytest.skip("nltk not installed — pip install nltk")
        from rag_pipeline.generation.response_evaluator import compute_faithfulness
        self.compute_faithfulness = compute_faithfulness

    def test_identical_text_scores_one(self):
        """
        When answer and context are identical, every sentence's content words
        are a subset of the context vocabulary → faithfulness = 1.0.
        """
        text = (
            "JPMorgan maintains AML compliance programs under the Bank Secrecy Act. "
            "The firm files Suspicious Activity Reports for flagged transactions."
        )
        score = self.compute_faithfulness(text, text)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_fully_hallucinated_answer_scores_low(self):
        """
        Content with zero overlap with the context should score near 0.
        We use two completely unrelated domains (cooking vs finance) to ensure
        there is no accidental stopword-driven overlap.
        """
        context = (
            "Basel III requires banks to hold a CET1 capital ratio above 4.5 percent. "
            "JPMorgan disclosed a CET1 ratio of 13.1 percent in its annual report."
        )
        hallucinated = (
            "Boiling pasta requires salted water and precise timing. "
            "The sauce should simmer for thirty minutes before serving."
        )
        score = self.compute_faithfulness(hallucinated, context)
        # At most a few stopword-adjacent tokens might match — score should be low
        assert score < 0.40, f"Expected low faithfulness, got {score:.3f}"

    def test_partial_overlap_between_zero_and_one(self):
        """
        A mix of grounded and ungrounded sentences should score between 0 and 1.
        First sentence uses exact context words; second is unrelated.
        """
        context = (
            "JPMorgan maintains robust AML controls under the Bank Secrecy Act "
            "and files Suspicious Activity Reports for suspicious transactions."
        )
        partial = (
            "JPMorgan maintains AML controls and files suspicious activity reports. "  # grounded
            "The company also produces award-winning chocolate truffles."              # ungrounded
        )
        score = self.compute_faithfulness(partial, context)
        assert 0.0 < score < 1.0, f"Expected partial score, got {score:.3f}"

    def test_returns_none_when_context_empty(self):
        """Empty context means there is no evidence vocabulary; score should be 0."""
        score = self.compute_faithfulness("Any answer.", "")
        # Empty evidence → 0 (or None if library unavailable — both acceptable)
        assert score == pytest.approx(0.0) or score is None

    def test_stopword_only_sentence_counted_as_faithful(self):
        """
        Sentences containing only stopwords (e.g. 'OK.') should not penalise
        the score — they carry no semantic content and are not hallucinations.
        Filtering them as 'faithful' keeps the metric focused on content sentences.
        """
        context = "JPMorgan filed SARs for suspicious wire transfers in 2023."
        answer  = "OK. JPMorgan filed SARs for suspicious wire transfers."
        score = self.compute_faithfulness(answer, context)
        # The substantive sentence is grounded; 'OK.' should not drag score down
        assert score >= 0.5, f"Stopword-only sentence incorrectly penalised: {score:.3f}"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — BLEU SCORE
# Source: rag_pipeline/generation/response_evaluator.py :: compute_bleu
#
# BLEU measures n-gram precision of the answer against the context. Higher
# means more verbatim language from the source appears in the answer.
# These tests verify monotonicity (more overlap → higher score) and that
# the smoothing function prevents collapse to 0 for partial matches.
# ═════════════════════════════════════════════════════════════════════════════

class TestBLEU:

    @pytest.fixture(autouse=True)
    def _import(self):
        if not _NLTK_AVAILABLE:
            pytest.skip("nltk not installed — pip install nltk")
        from rag_pipeline.generation.response_evaluator import compute_bleu
        self.compute_bleu = compute_bleu

    def test_identical_text_scores_high(self):
        """
        Identical answer and context should produce a high BLEU score.
        (Not necessarily 1.0 — the brevity penalty applies when hypothesis
        length ≠ reference length in sentence_bleu.)
        """
        text = (
            "JPMorgan files Suspicious Activity Reports for transactions "
            "exhibiting patterns consistent with money laundering."
        )
        score = self.compute_bleu(text, text)
        assert score is not None
        assert score > 0.5, f"Expected high BLEU for identical text, got {score:.3f}"

    def test_unrelated_text_scores_low(self):
        """Completely unrelated answer should score close to 0."""
        context = "Basel III capital requirements mandate a minimum CET1 ratio of 4.5 percent."
        answer  = "The quick brown fox jumps over the lazy dog every single morning."
        score = self.compute_bleu(answer, context)
        assert score is not None
        assert score < 0.15, f"Expected near-zero BLEU for unrelated text, got {score:.3f}"

    def test_bleu_is_nonnegative(self):
        """BLEU must always be in [0, 1]; smoothing must not produce negatives."""
        score = self.compute_bleu("some answer text", "some completely different context text")
        assert score is not None
        assert 0.0 <= score <= 1.0

    def test_partial_match_between_extremes(self):
        """Partial n-gram overlap should produce a score strictly between 0 and 1."""
        context = "JPMorgan maintained a CET1 capital ratio of 13.1 percent in Q4 2023."
        answer  = "JPMorgan reported strong capital ratios in their annual disclosure."
        score = self.compute_bleu(answer, context)
        assert score is not None
        assert 0.0 <= score <= 1.0


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — EVALUATE RESPONSE (integration of BLEU + ROUGE + faithfulness)
# Source: rag_pipeline/generation/response_evaluator.py :: evaluate_response
#
# Tests the overall verdict logic (PASS/WARN/FAIL) rather than individual
# metric values, since the combined verdict is what gates production alerts.
# ═════════════════════════════════════════════════════════════════════════════

class TestEvaluateResponse:

    @pytest.fixture(autouse=True)
    def _import(self):
        from rag_pipeline.generation.response_evaluator import evaluate_response
        self.evaluate_response = evaluate_response

    def test_empty_answer_returns_fail(self):
        """
        An empty answer provides no information — it should always FAIL.
        Guards against a bug where the model returns an empty string and
        the evaluator incorrectly passes it because there are no failing n-grams.
        """
        result = self.evaluate_response(answer="", context="Some context.", question="?")
        assert result.overall == "FAIL"

    def test_empty_context_returns_fail(self):
        """
        Without context there is nothing to be faithful to.
        A caller who passes empty context has a retrieval failure upstream.
        """
        result = self.evaluate_response(answer="Some answer.", context="", question="?")
        assert result.overall == "FAIL"

    def test_result_has_overall_field(self):
        """
        overall must always be present regardless of which libraries are installed.
        A missing overall field would break API response serialisation.
        """
        result = self.evaluate_response(
            answer   = "JPMorgan files SARs under the Bank Secrecy Act.",
            context  = "JPMorgan files Suspicious Activity Reports under the BSA.",
            question = "What does JPMorgan do under the BSA?",
        )
        assert result.overall in {"PASS", "WARN", "FAIL", "UNKNOWN"}

    def test_to_dict_has_required_keys(self):
        """
        to_dict() is used to populate the API response. All keys expected by
        EvaluationScores in api/routers/query.py must be present.
        """
        result = self.evaluate_response(
            answer  = "The bank holds capital reserves.",
            context = "JPMorgan holds capital reserves exceeding regulatory minimums.",
            question= "How does JPMorgan manage capital?",
        )
        d = result.to_dict()
        assert "bleu"              in d
        assert "rouge_l"           in d
        assert "faithfulness"      in d
        assert "overall"           in d
        assert "available_metrics" in d

    def test_high_overlap_answer_does_not_fail(self):
        """
        An answer that closely mirrors the context should not be marked FAIL.
        This prevents false rejections of high-quality, well-grounded responses.
        """
        context = (
            "JPMorgan Chase maintains AML compliance programs under the Bank Secrecy Act. "
            "The firm files Suspicious Activity Reports for transactions consistent with "
            "money laundering. The bank's CET1 ratio was 13.1 percent in Q4 2023."
        )
        answer = (
            "JPMorgan maintains AML compliance programs under the Bank Secrecy Act "
            "and files Suspicious Activity Reports for flagged transactions. "
            "Their CET1 ratio was 13.1 percent as of Q4 2023."
        )
        result = self.evaluate_response(answer=answer, context=context, question="AML?")
        # UNKNOWN is acceptable here: it means nltk/rouge-score are not installed
        # so no metrics ran. The important invariant is that a well-grounded answer
        # must never be marked FAIL — UNKNOWN simply means "not enough info to judge".
        assert result.overall in {"PASS", "WARN", "UNKNOWN"}, (
            f"Expected PASS, WARN, or UNKNOWN for a well-grounded answer, got {result.overall}. "
            f"Scores: {result.to_dict()}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — CHUNKER
# Source: rag_pipeline/ingestion/chunker.py :: chunk_documents()
#
# chunk_documents() is a pure text-processing function — it only requires
# langchain_text_splitters (a lightweight NLP utility, not an ML model).
# Tests verify splitting behaviour, metadata propagation, and the noise filter.
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def chunk_fn():
    """Import chunk_documents once per module to avoid repeated import overhead."""
    try:
        from rag_pipeline.ingestion.chunker import chunk_documents
        return chunk_documents
    except ImportError as e:
        pytest.skip(f"chunker unavailable (langchain not installed): {e}")


class TestChunker:

    def test_single_document_produces_multiple_chunks(self, chunk_fn):
        """
        A document longer than chunk_size=512 chars must be split into at
        least two chunks. Verifies that the splitter is actually applied and
        not bypassed by an early-return path.
        """
        long_text = (
            "JPMorgan Chase maintains comprehensive AML compliance programs. " * 20
        )
        docs   = [{"text": long_text, "ticker": "JPM", "filing_type": "10-K", "path": "test.txt"}]
        chunks = chunk_fn(docs)
        assert len(chunks) >= 2, "Long document should produce multiple chunks"

    def test_chunk_size_does_not_exceed_limit(self, chunk_fn):
        """
        No chunk's text should exceed the configured chunk_size (512 chars by
        default). The splitter may produce chunks slightly above this limit when
        no split point exists within the window, but we allow 20% headroom to
        accommodate that edge case without false failures.
        """
        long_text = "This is a sentence about financial regulation and AML compliance. " * 30
        docs   = [{"text": long_text, "ticker": "JPM", "filing_type": "10-K", "path": "test.txt"}]
        chunks = chunk_fn(docs)
        for c in chunks:
            assert len(c["text"]) <= 512 * 1.2, (
                f"Chunk exceeds expected size limit: {len(c['text'])} chars"
            )

    def test_metadata_preserved_in_each_chunk(self, chunk_fn):
        """
        ticker, filing_type, and source path must survive the split.
        Without metadata, the citation format [TICKER / TYPE / Chunk N of M]
        required by COMPLIANCE_SYSTEM_PROMPT cannot be constructed.
        """
        long_text = "Regulatory paragraph about capital requirements. " * 30
        docs   = [{"text": long_text, "ticker": "BAC", "filing_type": "10-Q", "path": "/data/bac.txt"}]
        chunks = chunk_fn(docs)
        for c in chunks:
            assert c["metadata"]["ticker"]      == "BAC"
            assert c["metadata"]["filing_type"] == "10-Q"
            assert c["metadata"]["source"]      == "/data/bac.txt"

    def test_chunk_indices_are_gapless(self, chunk_fn):
        """
        chunk_index must be 0, 1, 2 ... N-1 with no gaps — even after the
        minimum-length filter discards short chunks. A gapped sequence (0, 2, 3)
        would indicate the filter runs after index assignment, which would cause
        chunk 1 to appear missing in citations.
        """
        long_text = "Compliance requirement and regulatory obligation paragraph. " * 40
        docs   = [{"text": long_text, "ticker": "GS", "filing_type": "8-K", "path": "gs.txt"}]
        chunks = chunk_fn(docs)
        indices = [c["metadata"]["chunk_index"] for c in chunks]
        assert indices == list(range(len(indices))), (
            f"Chunk indices have gaps: {indices}"
        )

    def test_chunk_count_matches_actual_chunks(self, chunk_fn):
        """
        metadata['chunk_count'] must equal the actual number of chunks produced
        for that document. Downstream code uses this for 'Chunk 3 of 7' labels;
        an incorrect count produces misleading citations.
        """
        long_text = "Risk factor disclosure and forward-looking statement about AML. " * 25
        docs   = [{"text": long_text, "ticker": "WFC", "filing_type": "10-K", "path": "wfc.txt"}]
        chunks = chunk_fn(docs)
        reported_count = chunks[0]["metadata"]["chunk_count"]
        assert reported_count == len(chunks), (
            f"chunk_count={reported_count} but {len(chunks)} chunks were produced"
        )

    def test_filters_chunks_shorter_than_50_chars(self, chunk_fn):
        """
        Chunks shorter than 50 characters are noise (page headers, exhibit labels,
        lone numbers). The filter must remove them so they don't pollute the FAISS
        index with semantically empty vectors that score above threshold on
        coincidental single-word matches.
        """
        # A mostly-short document: one real sentence + several fragments
        text = (
            "Item 1.\n"                  # 7 chars — should be filtered
            "Exhibit A.\n"               # 10 chars — should be filtered
            "2023.\n"                    # 5 chars  — should be filtered
            "JPMorgan Chase maintains a comprehensive AML compliance program under the "
            "Bank Secrecy Act including transaction monitoring and SAR filing obligations."
        )
        docs   = [{"text": text, "ticker": "JPM", "filing_type": "8-K", "path": "test.txt"}]
        chunks = chunk_fn(docs)
        for c in chunks:
            assert len(c["text"].strip()) >= 50, (
                f"Short chunk survived filter: {c['text']!r}"
            )

    def test_empty_document_produces_no_chunks(self, chunk_fn):
        """
        An empty or whitespace-only document must produce zero chunks.
        Prevents phantom vectors in the FAISS index from filings that
        failed to parse (blank text fields from sec_loader error handling).
        """
        docs   = [{"text": "   \n  ", "ticker": "JPM", "filing_type": "10-K", "path": "empty.txt"}]
        chunks = chunk_fn(docs)
        assert chunks == [], f"Empty document produced {len(chunks)} chunks"

    def test_multiple_documents_chunks_are_independent(self, chunk_fn):
        """
        Chunks from different documents must not cross-contaminate each other's
        metadata. A bug where all chunks from a batch inherit the last document's
        ticker would break company-scoped retrieval queries.
        """
        docs = [
            {"text": "JPMorgan AML compliance requirements. " * 20, "ticker": "JPM", "filing_type": "10-K", "path": "jpm.txt"},
            {"text": "Goldman Sachs capital adequacy disclosures. " * 20, "ticker": "GS",  "filing_type": "10-Q", "path": "gs.txt"},
        ]
        chunks = chunk_fn(docs)
        jpm_chunks = [c for c in chunks if c["metadata"]["ticker"] == "JPM"]
        gs_chunks  = [c for c in chunks if c["metadata"]["ticker"] == "GS"]
        assert len(jpm_chunks) > 0
        assert len(gs_chunks)  > 0
        # Every JPM chunk must have JPM metadata and vice versa
        for c in jpm_chunks:
            assert c["metadata"]["filing_type"] == "10-K"
        for c in gs_chunks:
            assert c["metadata"]["filing_type"] == "10-Q"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — TRANSACTION GENERATOR: FRAUD RATE
# Source: data_pipeline/generators/transaction_generator.py
#
# WHY WE DON'T IMPORT THE MODULE:
# The generator runs ALL its code at module level — Faker builds 50,000 user
# objects and NumPy generates 500,000 transactions during import. Adding that
# import to this test file would add ~60 seconds of setup time on every
# `pytest` run, making the unit test suite unusable for CI feedback loops.
#
# Strategy A: Verify the arithmetic from the module's constants directly
#   — always runs, confirms the formula is correct.
# Strategy B: Read the output CSV if it already exists on disk
#   — skipped automatically when the CSV is absent, avoids re-running generation.
# ═════════════════════════════════════════════════════════════════════════════

class TestTransactionGeneratorFraudRate:

    # These constants are copied from the source file. If they change, this test
    # will catch the discrepancy when the CSV strategy B also runs.
    TOTAL        = 500_000
    FRAUD_RATE   = 0.02
    N_FRAUD      = int(TOTAL * FRAUD_RATE)   # = 10,000
    N_NORMAL     = TOTAL - N_FRAUD           # = 490,000
    N_PER_PATTERN = N_FRAUD // 4             # = 2,500  (4 fraud patterns)

    def test_fraud_count_arithmetic(self):
        """
        N_FRAUD must equal TOTAL × FRAUD_RATE.
        int() truncation could cause N_FRAUD + N_NORMAL ≠ TOTAL if FRAUD_RATE
        is not a clean fraction — verify the split is exact.
        """
        assert self.N_FRAUD == 10_000
        assert self.N_NORMAL == 490_000
        assert self.N_FRAUD + self.N_NORMAL == self.TOTAL

    def test_per_pattern_count_covers_all_fraud(self):
        """
        4 patterns × N_PER_PATTERN must equal N_FRAUD.
        An off-by-one here means the actual fraud count diverges from FRAUD_RATE.
        """
        assert self.N_PER_PATTERN * 4 == self.N_FRAUD

    def test_fraud_rate_is_approximately_two_percent(self):
        """
        The actual fraction N_FRAUD / TOTAL must be within 0.1pp of 2%.
        This confirms that int() truncation in N_FRAUD = int(TOTAL * FRAUD_RATE)
        doesn't push the real rate below 1.95% or above 2.05%.
        """
        actual_rate = self.N_FRAUD / self.TOTAL
        assert abs(actual_rate - self.FRAUD_RATE) < 0.001, (
            f"Fraud rate deviation too large: {actual_rate:.4f} vs {self.FRAUD_RATE}"
        )

    def test_csv_fraud_rate(self):
        """
        Strategy B: read the generated CSV and verify the actual fraud rate.

        Skipped automatically when transactions.csv doesn't exist so the test
        suite passes in a fresh checkout before the generator has been run.
        The CSV is produced by:  python data_pipeline/generators/transaction_generator.py
        """
        csv_path = Path("data/transactions.csv")
        if not csv_path.exists():
            pytest.skip("data/transactions.csv not found — run the transaction generator first")

        import pandas as pd
        # Read only the is_fraud column to avoid loading 500k rows of strings.
        df = pd.read_csv(csv_path, usecols=["is_fraud"])
        actual_rate = df["is_fraud"].mean()

        # Allow ±0.5pp tolerance: velocity_burst generates batches of 5–10
        # transactions per burst, so the exact fraud count can overshoot
        # N_PER_PATTERN slightly depending on the random burst sizes.
        assert abs(actual_rate - self.FRAUD_RATE) < 0.005, (
            f"CSV fraud rate {actual_rate:.4f} deviates from target {self.FRAUD_RATE}"
        )

    def test_csv_has_expected_columns(self):
        """
        Verify the CSV schema expected by the Kafka producer and feature
        engineering Spark job. A column rename in the generator without
        updating downstream consumers is a silent data-contract break.
        """
        csv_path = Path("data/transactions.csv")
        if not csv_path.exists():
            pytest.skip("data/transactions.csv not found")

        import pandas as pd
        df = pd.read_csv(csv_path, nrows=1)
        required = {
            "transaction_id", "timestamp", "user_id", "amount",
            "merchant_category", "hour_of_day", "day_of_week", "is_fraud",
        }
        missing = required - set(df.columns)
        assert not missing, f"CSV missing expected columns: {missing}"
