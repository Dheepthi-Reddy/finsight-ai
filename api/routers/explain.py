"""
SHAP explainability endpoints for the FinSight fraud detection model.

What these endpoints are used for
-----------------------------------
The fraud scoring endpoint (POST /fraud/score) is a black box from the
caller's perspective: it returns a number and a risk tier but not a reason.
These explanation endpoints open up that black box.

  GET /explain/{transaction_id}
  ─────────────────────────────
  Primary use case: a fraud analyst or customer service agent is reviewing a
  flagged transaction and needs to know *why* it was flagged before deciding
  to block or approve it. The transaction was already scored by POST /fraud/score;
  its feature values are stored in FraudPredictor's in-memory store. This
  endpoint retrieves those features and runs SHAP TreeExplainer to produce
  per-feature Shapley values, ranked by impact.

  GET /explain/importance
  ────────────────────────
  Use case: model monitoring dashboards and quarterly model reviews. Returns
  global feature importance — the average contribution of each feature across
  the training distribution. Useful for:
    • Detecting feature drift (a feature that mattered in training now scores
      near zero, suggesting its values have shifted in production).
    • Regulatory explainability reports that must describe which signals drive
      the model's decisions at a population level, not just for one transaction.
    • Debugging: if amount_x_risk suddenly drops in importance it may indicate
      a data pipeline issue upstream.

  POST /explain/inline
  ─────────────────────
  Use case: hypothetical what-if analysis without storing a transaction. An
  analyst can post any feature set and get back an instant SHAP breakdown:
    • "What score would this transaction get if the amount were $10,000 instead?"
    • "How would switching from grocery to wire_transfer change the risk?"
    • Testing the model interactively during development before wiring up the
      full scoring pipeline.

Route registration order
------------------------
FastAPI matches routes in registration order. `/explain/importance` must be
registered BEFORE `/explain/{transaction_id}` — otherwise the literal string
"importance" would be captured as a transaction_id path parameter and the
endpoint would never match.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator, model_validator

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# SHAP ANALYZER ACCESSOR
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_analyzer():
    """
    Load and cache a SHAPAnalyzer singleton.

    SHAPAnalyzer loads the same fraud_model.joblib and shap_explainer.joblib
    artifacts as FraudPredictor. In a memory-constrained environment you could
    share the explainer object between the two classes; for clarity we keep them
    independent here and let the OS page-cache absorb the duplicate file reads.

    lru_cache ensures the 50 MB of model weights is loaded exactly once per
    process regardless of how many explain requests arrive concurrently.
    """
    from fraud_model.evaluation.shap_analysis import SHAPAnalyzer
    return SHAPAnalyzer()


def _resolve_predictor(request: Request):
    """Return the predictor from app.state (set by lifespan in main.py)."""
    return getattr(request.app.state, "predictor", None)


def _resolve_analyzer():
    """Return the SHAPAnalyzer, creating it lazily via lru_cache if needed."""
    return _get_analyzer()


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# Mirrors fraud.py MERCHANT_RISK — kept here to engineer features for the
# inline endpoint without importing from the fraud router.
_MERCHANT_RISK: dict[str, float] = {
    "grocery":        0.05,
    "restaurant":     0.05,
    "gas_station":    0.10,
    "retail":         0.10,
    "pharmacy":       0.05,
    "entertainment":  0.10,
    "travel":         0.20,
    "subscription":   0.05,
    "utilities":      0.05,
    "healthcare":     0.05,
    "pawn_shop":      0.90,
    "wire_transfer":  0.95,
    "crypto_exchange":0.90,
    "money_order":    0.85,
    "check_cashing":  0.85,
}


def _reverse_category_mapping(predictor) -> dict[int, str]:
    """Build int → category_name from predictor's str → int mapping."""
    return {v: k for k, v in predictor.category_mapping.items()}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ─────────────────────────────────────────────────────────────────────────────
# SHARED RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────────

class DriverDetail(BaseModel):
    """
    Single feature's contribution to the fraud prediction.

    shap_value is in log-odds space. Positive means this feature pushed the
    prediction toward fraud; negative means it pushed toward legitimate.
    The magnitude indicates how strongly: a SHAP of +2.4 is roughly 4× more
    impactful than a SHAP of +0.6.
    """
    rank:          int   = Field(description="1-based rank by absolute SHAP value. Rank 1 = most impactful.")
    feature:       str   = Field(description="Feature name.")
    feature_value: Any   = Field(description="Value of this feature for the scored transaction.")
    shap_value:    float = Field(description="Shapley value in log-odds. Positive = toward fraud.")
    direction:     str   = Field(
        description="'increases_fraud_risk' if shap_value > 0, else 'decreases_fraud_risk'.",
        examples=["increases_fraud_risk"],
    )


class ExplainResponse(BaseModel):
    """SHAP explanation for one transaction."""

    transaction_id:   str   = Field(description="Transaction identifier.")
    fraud_probability:float = Field(description="Model probability that this transaction is fraud.")
    risk_level:       str   = Field(description="Risk tier: LOW / MEDIUM / HIGH / CRITICAL.")
    predicted_label:  int   = Field(description="Hard label at the F1-optimal threshold: 1=fraud, 0=legitimate.")
    scored_at:        str | None = Field(description="ISO-8601 UTC timestamp of original scoring, if available.")

    base_value:       float = Field(
        description="Log-odds base value (model's expected output over training data).",
    )
    base_fraud_rate:  float = Field(
        description=(
            "sigmoid(base_value) — the unconditional fraud probability implied by the "
            "training distribution. Typically ~2% for card transaction datasets."
        ),
    )
    top_drivers:      list[DriverDetail] = Field(
        description="Top 5 features ranked by absolute SHAP value, most impactful first.",
    )
    all_shap_values:  dict[str, float] = Field(
        description="Complete per-feature SHAP values in log-odds for all model features.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 1 — GET /explain/importance  (registered BEFORE /{transaction_id})
# ─────────────────────────────────────────────────────────────────────────────

class FeatureImportanceItem(BaseModel):
    rank:       int   = Field(description="1-based rank. Rank 1 = highest importance.")
    feature:    str   = Field(description="Feature name.")
    importance: float = Field(description="Normalised importance score in [0, 1]. All scores sum to 1.")
    importance_pct: str = Field(description="Importance as a percentage string, e.g. '34.7%'.")


class GlobalImportanceResponse(BaseModel):
    """
    Global feature importance derived from the trained XGBoost model.

    importance_type 'gain' measures the average reduction in training loss
    (cross-entropy) each feature provides across all tree splits where it
    appears. It answers: which features did the model find most useful overall?

    This is distinct from SHAP-based importance (mean |SHAP|), which measures
    the average *magnitude* of a feature's prediction contribution at inference
    time. Gain is computed from training; mean |SHAP| is computed over a
    sample of actual predictions. Both tell similar stories for well-trained
    models; gain is used here because it requires no sample data and is instant.
    """
    importance_type: str                    = Field(description="Metric used: 'gain' (XGBoost split gain).")
    features:        list[FeatureImportanceItem] = Field(description="Features ranked by importance.")
    model_threshold: float                  = Field(description="F1-optimal classification threshold from training.")


@router.get(
    "/importance",
    response_model = GlobalImportanceResponse,
    status_code    = status.HTTP_200_OK,
    summary        = "Global feature importance",
    description    = (
        "Returns XGBoost gain-based feature importance for all model features, "
        "normalised to sum to 1. Useful for model monitoring and regulatory explainability reports."
    ),
)
async def global_importance(request: Request) -> GlobalImportanceResponse:
    """
    Return XGBoost gain-based feature importance for the fraud model.

    No sample data or SHAP computation required — reads directly from the
    trained model's internal split statistics. Always fast (<1 ms).
    """
    predictor = _resolve_predictor(request)
    if predictor is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = "Fraud model is not loaded. Check GET /health/ready.",
        )

    try:
        # get_booster().get_score() returns {feature_name: gain_value} for all
        # features that appear in at least one tree split. Features that were
        # never used (zero importance) are absent; we add them explicitly as 0.
        raw_scores: dict[str, float] = predictor.model.get_booster().get_score(
            importance_type="gain"
        )
    except AttributeError:
        # Fallback for sklearn-API models: .feature_importances_ is a numpy
        # array in the same order as feature_cols.
        import numpy as np
        raw_array = predictor.model.feature_importances_
        raw_scores = dict(zip(predictor.feature_cols, raw_array.tolist()))

    # Ensure all feature_cols are present, even those with zero gain.
    all_scores = {f: raw_scores.get(f, 0.0) for f in predictor.feature_cols}

    total = sum(all_scores.values()) or 1.0   # guard against all-zero
    normalised = {f: v / total for f, v in all_scores.items()}

    ranked = sorted(normalised.items(), key=lambda x: x[1], reverse=True)
    items  = [
        FeatureImportanceItem(
            rank           = i + 1,
            feature        = feat,
            importance     = round(score, 6),
            importance_pct = f"{score * 100:.1f}%",
        )
        for i, (feat, score) in enumerate(ranked)
    ]

    return GlobalImportanceResponse(
        importance_type = "gain",
        features        = items,
        model_threshold = round(predictor.threshold, 4),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 2 — GET /explain/{transaction_id}
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{transaction_id}",
    response_model = ExplainResponse,
    status_code    = status.HTTP_200_OK,
    summary        = "Explain a previously scored transaction",
    description    = (
        "Retrieves stored feature values for a transaction scored by POST /fraud/score, "
        "runs SHAP TreeExplainer, and returns per-feature Shapley values with the "
        "top 5 drivers ranked by impact magnitude."
    ),
)
async def explain_transaction(
    transaction_id: str,
    request:        Request,
) -> ExplainResponse:
    """
    Explain why a transaction received its fraud score.

    Requires that POST /fraud/score was called for this transaction_id within
    the current server session. Scored transactions are held in FraudPredictor's
    in-memory store (up to 100,000 entries). Transactions evicted from the store
    return 404 — re-submit the transaction to POST /fraud/score first.

    SHAP computation takes ~5–15 ms for a single transaction using XGBoost's
    fast TreeExplainer path (exact Shapley values, no approximation).
    """
    predictor = _resolve_predictor(request)
    if predictor is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = "Fraud model is not loaded. Check GET /health/ready.",
        )

    # Retrieve the stored ScoreResult (contains pre-encoded features).
    stored = predictor.get_transaction(transaction_id)
    if stored is None:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = (
                f"Transaction '{transaction_id}' not found in the scoring store. "
                "Either it was never scored, or it was evicted after the store reached "
                "its 100,000-entry capacity. Re-submit the transaction to "
                "POST /fraud/score to generate a fresh score and store entry."
            ),
        )

    # Build a reverse mapping so we can show the human-readable merchant
    # category in the feature_value column instead of its integer encoding.
    reverse_cat = _reverse_category_mapping(predictor)

    # Convert stored encoded features to the format SHAPAnalyzer.explain() expects.
    # features dict has merchant_category as an int; _encode_transaction() in
    # SHAPAnalyzer handles both str and int, so we can pass it directly.
    # We swap the int back to a string for display purposes only.
    display_features = dict(stored.features)
    if isinstance(display_features.get("merchant_category"), (int, float)):
        code = int(display_features["merchant_category"])
        display_features["merchant_category"] = reverse_cat.get(code, str(code))

    try:
        analyzer     = _resolve_analyzer()
        explanation  = analyzer.explain(display_features)
    except Exception as exc:
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = f"SHAP explanation failed: {exc}",
        ) from exc

    drivers = [
        DriverDetail(
            rank          = i + 1,
            feature       = d["feature"],
            feature_value = d["feature_value"],
            shap_value    = round(d["shap_value"], 6),
            direction     = d["direction"],
        )
        for i, d in enumerate(explanation.top_5_drivers)
    ]

    return ExplainResponse(
        transaction_id    = stored.transaction_id,
        fraud_probability = round(stored.fraud_probability, 4),
        risk_level        = stored.risk_level,
        predicted_label   = explanation.predicted_label,
        scored_at         = stored.scored_at,
        base_value        = round(explanation.base_value, 6),
        base_fraud_rate   = round(_sigmoid(explanation.base_value), 6),
        top_drivers       = drivers,
        all_shap_values   = {k: round(v, 6) for k, v in explanation.shap_values.items()},
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 3 — POST /explain/inline
# ─────────────────────────────────────────────────────────────────────────────

class InlineExplainRequest(BaseModel):
    """
    Raw transaction fields for on-the-fly SHAP explanation without prior scoring.

    Identical contract to POST /fraud/score — the API engineers all derived
    features internally. Useful for what-if analysis:
      "How does the score change if I raise the amount to $10,000?"
      "What if this happened at a grocery store instead of a wire transfer?"
    """
    amount:             float = Field(..., ge=0.0, examples=[3200.00])
    merchant_category:  str   = Field(..., examples=["wire_transfer"])
    hour_of_day:        int   = Field(..., ge=0, le=23, examples=[2])
    day_of_week:        int   = Field(..., ge=0, le=6,  examples=[6])
    account_age_bucket: int   = Field(default=2, ge=0, le=2)
    txn_count_1h:       int   = Field(default=1, ge=1)
    amount_sum_1h:      float | None = Field(default=None, ge=0.0)

    @field_validator("merchant_category")
    @classmethod
    def validate_merchant_category(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in _MERCHANT_RISK:
            raise ValueError(
                f"Unknown merchant_category '{v}'. "
                f"Valid values: {sorted(_MERCHANT_RISK)}"
            )
        return v

    @model_validator(mode="after")
    def default_amount_sum(self) -> "InlineExplainRequest":
        if self.amount_sum_1h is None:
            self.amount_sum_1h = self.amount
        return self

    model_config = {"json_schema_extra": {
        "example": {
            "amount":            3200.00,
            "merchant_category": "wire_transfer",
            "hour_of_day":       2,
            "day_of_week":       6,
            "account_age_bucket":0,
            "txn_count_1h":      6,
            "amount_sum_1h":     9800.00,
        }
    }}


class InlineExplainResponse(BaseModel):
    """SHAP explanation for an ad-hoc feature set. No transaction_id or stored entry."""
    fraud_probability: float
    risk_level:        str
    predicted_label:   int
    base_value:        float
    base_fraud_rate:   float
    top_drivers:       list[DriverDetail]
    all_shap_values:   dict[str, float]
    features_used:     dict[str, Any] = Field(
        description="The engineered feature values that were explained.",
    )


@router.post(
    "/inline",
    response_model = InlineExplainResponse,
    status_code    = status.HTTP_200_OK,
    summary        = "Explain any transaction without prior scoring",
    description    = (
        "Accepts raw transaction fields, engineers model features, scores the "
        "transaction, and returns a full SHAP breakdown — all in one call. "
        "Does not persist anything to the scoring store. Ideal for what-if analysis."
    ),
)
async def explain_inline(
    request: Request,
    body:    InlineExplainRequest,
) -> InlineExplainResponse:
    """
    Score and explain a transaction in a single call.

    Unlike GET /explain/{transaction_id}, this endpoint does not require a prior
    POST /fraud/score call. It engineers features, runs inference, and runs SHAP
    in one request. Use it for interactive exploration or UI-driven what-if sliders.

    The scored result is NOT stored in FraudPredictor's in-memory store, so
    this transaction_id will not be retrievable via GET /explain/{transaction_id}.
    """
    predictor = _resolve_predictor(request)
    if predictor is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = "Fraud model is not loaded. Check GET /health/ready.",
        )

    # ── Engineer features (same logic as fraud.py) ────────────────────────────
    risk_weight = _MERCHANT_RISK[body.merchant_category]
    features: dict[str, Any] = {
        "amount":             body.amount,
        "amount_log":         round(math.log1p(body.amount), 6),
        "amount_x_risk":      round(body.amount * risk_weight, 4),
        "amount_sum_1h":      body.amount_sum_1h,
        "txn_count_1h":       body.txn_count_1h,
        "hour_of_day":        body.hour_of_day,
        "day_of_week":        body.day_of_week,
        "is_night":           1 if body.hour_of_day <= 5 else 0,
        "is_weekend":         1 if body.day_of_week >= 5 else 0,
        "account_age_bucket": body.account_age_bucket,
        "merchant_category":  body.merchant_category,
    }

    # ── Score + explain ───────────────────────────────────────────────────────
    try:
        score_result = predictor.score(features)
        analyzer     = _resolve_analyzer()
        explanation  = analyzer.explain(features)
    except ValueError as exc:
        raise HTTPException(
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail      = f"Explanation failed — invalid input: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = f"Explanation failed — internal error: {exc}",
        ) from exc

    drivers = [
        DriverDetail(
            rank          = i + 1,
            feature       = d["feature"],
            feature_value = d["feature_value"],
            shap_value    = round(d["shap_value"], 6),
            direction     = d["direction"],
        )
        for i, d in enumerate(explanation.top_5_drivers)
    ]

    return InlineExplainResponse(
        fraud_probability = round(score_result.fraud_probability, 4),
        risk_level        = score_result.risk_level,
        predicted_label   = explanation.predicted_label,
        base_value        = round(explanation.base_value, 6),
        base_fraud_rate   = round(_sigmoid(explanation.base_value), 6),
        top_drivers       = drivers,
        all_shap_values   = {k: round(v, 6) for k, v in explanation.shap_values.items()},
        features_used     = features,
    )
