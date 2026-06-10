"""
Fraud scoring endpoints for the FinSight API.

The caller sends a raw transaction (amount, merchant, time of day, etc.) and
receives a fraud probability, risk tier, and the engineered feature values
that were actually fed to the model. Feature engineering is done here, inside
the API, so callers never need to know the model's internal representation.

Why engineer features in the API layer?
----------------------------------------
The XGBoost model was trained on derived features (amount_log, amount_x_risk,
is_night, etc.), not on the raw fields a payment processor naturally has.
Moving that logic into the API means:
  - Callers send the natural transaction representation they already have.
  - Feature logic lives in one place — no risk of the caller computing
    amount_log with log() while the model was trained with log1p().
  - The response includes features_used so analysts can audit exactly what
    the model received, without access to internal code.

Velocity features (txn_count_1h, amount_sum_1h)
-------------------------------------------------
These count how many transactions a user made and their total spend in the
preceding hour. Computing them correctly requires a sliding window over the
user's transaction history — typically maintained in Redis or a Flink/Spark
Streaming job. The API accepts them as optional inputs so callers can pass the
authoritative values from their own streaming layer. When absent, conservative
defaults are used (1 transaction, this amount only) with a flag in the
response so downstream consumers know the velocity signal was approximate.
"""

from __future__ import annotations

import math
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator, model_validator

router = APIRouter()

# ── Merchant risk lookup ───────────────────────────────────────────────────────
# Mirrors data_pipeline/spark/feature_engineering.py MERCHANT_RISK exactly.
# amount_x_risk = amount × risk_factor; high-risk merchants amplify the amount
# signal so large wire transfers score very differently from large grocery runs.
# Kept here rather than imported from the data pipeline to avoid a cross-module
# dependency between the API and a Spark job.
MERCHANT_RISK: dict[str, float] = {
    "grocery": 0.05,
    "restaurant": 0.05,
    "gas_station": 0.10,
    "retail": 0.10,
    "pharmacy": 0.05,
    "entertainment": 0.10,
    "travel": 0.20,
    "subscription": 0.05,
    "utilities": 0.05,
    "healthcare": 0.05,
    "pawn_shop": 0.90,
    "wire_transfer": 0.95,
    "crypto_exchange": 0.90,
    "money_order": 0.85,
    "check_cashing": 0.85,
}

VALID_MERCHANT_CATEGORIES = sorted(MERCHANT_RISK.keys())

# Buckets mirror account_age_bucket from feature_engineering.py:
#   0 = new account       (< 90 days since first observed transaction)
#   1 = mid-age account   (90–364 days)
#   2 = established account (≥ 365 days)
ACCOUNT_AGE_BUCKET_MAP = {0: "< 90 days", 1: "90–364 days", 2: "≥ 365 days"}


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST MODEL
# ─────────────────────────────────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    """
    Raw transaction fields a payment processor naturally holds.

    The API derives all engineered model features from these inputs, so callers
    never need to know about amount_log, amount_x_risk, or is_night.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    # Optional — if absent, the API generates a UUID. Providing an idempotent
    # ID allows callers to re-score the same transaction without creating
    # duplicate entries in the in-memory store.
    transaction_id: str | None = Field(
        default=None,
        description="Caller-supplied transaction ID. A UUID is generated if absent.",
        examples=["txn-8821"],
    )

    # ── Amount ────────────────────────────────────────────────────────────────
    # Raw transaction value in USD. $0 is valid (card authorisation hold).
    # gt=-0.01 allows $0 authorisations while blocking negative values.
    amount: float = Field(
        ...,
        ge=0.0,
        description="Transaction amount in USD.",
        examples=[3200.00],
    )

    # ── Merchant ──────────────────────────────────────────────────────────────
    # Must be one of VALID_MERCHANT_CATEGORIES — validated below.
    # The risk weight associated with the category is the primary driver of
    # amount_x_risk, the model's strongest fraud signal.
    merchant_category: str = Field(
        ...,
        description=f"Merchant category. Valid values: {VALID_MERCHANT_CATEGORIES}",
        examples=["wire_transfer"],
    )

    # ── Time ─────────────────────────────────────────────────────────────────
    # hour_of_day 0–23: transactions at 2–4 AM (is_night=1) have higher fraud
    # rates because most legitimate cardholders are asleep.
    # day_of_week 0–6 (0=Monday): weekend transactions (is_weekend=1) at
    # high-risk merchants show elevated fraud in training data.
    hour_of_day: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour of the transaction in 24h format (0–23, local merchant time).",
        examples=[2],
    )
    day_of_week: int = Field(
        ...,
        ge=0,
        le=6,
        description="Day of week: 0=Monday … 6=Sunday.",
        examples=[6],
    )

    # ── Account age ───────────────────────────────────────────────────────────
    # New accounts are disproportionately used for fraud — criminals open them
    # specifically for a single fraudulent transaction then abandon them.
    # Callers should derive the bucket from their own account-creation date:
    #   0 if days_since_creation < 90
    #   1 if 90 <= days_since_creation < 365
    #   2 otherwise
    account_age_bucket: int = Field(
        default=2,
        ge=0,
        le=2,
        description=(
            "Account age tier: 0=new (<90d), 1=mid (90–364d), 2=established (≥365d). "
            "Defaults to 2 (established) when unknown — conservative assumption."
        ),
        examples=[0],
    )

    # ── Velocity features (optional) ──────────────────────────────────────────
    # Velocity signals are the second-strongest fraud indicator after
    # amount_x_risk. A burst of 6 transactions in an hour at 2 AM from a new
    # account is a textbook card-testing pattern.
    #
    # Callers should populate these from a Redis sorted-set or streaming
    # aggregation (Flink, Spark Structured Streaming) that maintains a sliding
    # window per user_id. If absent, the API defaults to:
    #   txn_count_1h  = 1  (this transaction only — conservative low estimate)
    #   amount_sum_1h = amount (only this transaction counted)
    txn_count_1h: int = Field(
        default=1,
        ge=1,
        description=(
            "Number of transactions by this user in the preceding 60 minutes, "
            "including this one. Provide from your streaming layer for best accuracy."
        ),
        examples=[6],
    )
    amount_sum_1h: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Total USD amount spent by this user in the preceding 60 minutes, "
            "including this transaction. Defaults to `amount` when absent."
        ),
        examples=[9800.00],
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("merchant_category")
    @classmethod
    def validate_merchant_category(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in MERCHANT_RISK:
            raise ValueError(
                f"Unknown merchant_category '{v}'. "
                f"Valid values: {VALID_MERCHANT_CATEGORIES}"
            )
        return v

    @model_validator(mode="after")
    def default_amount_sum_1h(self) -> "TransactionRequest":
        """Default amount_sum_1h to amount when not supplied."""
        if self.amount_sum_1h is None:
            self.amount_sum_1h = self.amount
        return self

    model_config = {"json_schema_extra": {
        "example": {
            "transaction_id": "txn-8821",
            "amount": 3200.00,
            "merchant_category": "wire_transfer",
            "hour_of_day": 2,
            "day_of_week": 6,
            "account_age_bucket": 0,
            "txn_count_1h": 6,
            "amount_sum_1h": 9800.00,
        }
    }}


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE MODEL
# ─────────────────────────────────────────────────────────────────────────────

class FeaturesUsed(BaseModel):
    """
    The engineered feature values passed to the model.

    Included in the response so analysts and downstream systems can audit
    exactly what the model received, without reading the API source code.
    """
    amount: float = Field(description="Raw transaction amount in USD.")
    amount_log: float = Field(
        description="log1p(amount) — compresses the skewed amount distribution.")
    amount_x_risk: float = Field(
        description="amount × merchant risk weight — the model's primary fraud signal.")
    amount_sum_1h: float = Field(description="Total spend by this user in the preceding hour.")
    txn_count_1h: int = Field(
        description="Number of transactions by this user in the preceding hour.")
    hour_of_day: int = Field(description="Hour of the transaction (0–23).")
    day_of_week: int = Field(description="Day of the week (0=Monday, 6=Sunday).")
    is_night: int = Field(description="1 if hour_of_day is 0–5 (midnight to 5:59 AM), else 0.")
    is_weekend: int = Field(description="1 if day_of_week is Saturday (5) or Sunday (6), else 0.")
    account_age_bucket: int = Field(description="Account age tier: 0=new, 1=mid, 2=established.")
    merchant_category: str = Field(description="Raw merchant category string (pre-encoding).")
    velocity_estimated: bool = Field(
        description=(
            "True if txn_count_1h or amount_sum_1h were not supplied by the caller "
            "and were defaulted to conservative estimates. "
            "Scores with velocity_estimated=True are less accurate for burst-pattern fraud."
        )
    )


class ScoreResponse(BaseModel):
    """Fraud scoring result for a single transaction."""

    # ── Core prediction ───────────────────────────────────────────────────────
    transaction_id: str = Field(
        description="Transaction ID — caller-supplied or auto-generated UUID."
    )
    fraud_probability: float = Field(
        description="Model output: probability that this transaction is fraudulent. Range [0, 1].",
        examples=[0.9132],
    )
    is_fraud: bool = Field(
        description=(
            "Hard fraud label at the F1-optimal threshold from training. "
            "Use fraud_probability for nuanced downstream decisions; use is_fraud "
            "for binary routing (block vs allow)."
        ),
    )
    risk_level: str = Field(
        description=(
            "Actionable risk tier derived from fraud_probability: "
            "LOW (<30%) — allow; "
            "MEDIUM (30–60%) — soft review; "
            "HIGH (60–85%) — manual review queue; "
            "CRITICAL (≥85%) — block immediately."
        ),
        examples=["CRITICAL"],
    )
    scored_at: str = Field(
        description="ISO-8601 UTC timestamp of when scoring completed.",
    )

    # ── Transparency ──────────────────────────────────────────────────────────
    features_used: FeaturesUsed = Field(
        description="Engineered feature values that were passed to the model."
    )


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def _engineer_features(req: TransactionRequest) -> tuple[dict[str, Any], bool]:
    """
    Derive all model input features from the raw TransactionRequest.

    Returns (feature_dict, velocity_estimated) where velocity_estimated is
    True when the caller did not supply txn_count_1h / amount_sum_1h.

    Features and their fraud signal:
      amount_log      — log1p compresses the right-skewed amount distribution
                        so the model treats $5,000 and $50,000 more similarly
                        than $50 and $500. log1p instead of log because log(0)
                        is undefined (some transactions are $0 authorisations).
      amount_x_risk   — amount multiplied by the merchant's risk weight.
                        A $3,200 wire transfer (risk=0.95) scores 3,040; the
                        same amount at a grocery store (risk=0.05) scores 160.
                        This single feature drives the largest SHAP values.
      is_night        — hours 0–5 are treated as night. Fraud disproportionately
                        occurs when the cardholder is asleep and less likely to
                        notice or dispute an alert.
      is_weekend      — weekend + high-risk merchant is a common fraud pattern
                        because bank fraud teams are smaller on weekends.
      txn_count_1h,   — velocity signals: bursts of rapid transactions are the
      amount_sum_1h     primary indicator of card-testing and account takeover.
      account_age_bucket — new accounts (bucket 0) are disproportionately
                        opened specifically for a single fraudulent use.
    """
    # Track whether velocity was estimated so we can flag it in the response.
    # The request validator already defaulted amount_sum_1h to amount, so we
    # compare against the original request defaults.
    velocity_estimated = (req.txn_count_1h == 1 and req.amount_sum_1h == req.amount)

    risk_weight = MERCHANT_RISK[req.merchant_category]
    amount_log = round(math.log1p(req.amount), 6)
    amount_x_risk = round(req.amount * risk_weight, 4)
    is_night = 1 if req.hour_of_day <= 5 else 0
    is_weekend = 1 if req.day_of_week >= 5 else 0

    features: dict[str, Any] = {
        "amount": req.amount,
        "amount_log": amount_log,
        "amount_x_risk": amount_x_risk,
        "amount_sum_1h": req.amount_sum_1h,
        "txn_count_1h": req.txn_count_1h,
        "hour_of_day": req.hour_of_day,
        "day_of_week": req.day_of_week,
        "is_night": is_night,
        "is_weekend": is_weekend,
        "account_age_bucket": req.account_age_bucket,
        "merchant_category": req.merchant_category,
    }
    if req.transaction_id:
        features["transaction_id"] = req.transaction_id

    return features, velocity_estimated


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/score",
    response_model=ScoreResponse,
    status_code=status.HTTP_200_OK,
    summary="Score a transaction for fraud",
    description=(
        "Accepts a raw transaction, engineers model features, and returns a "
        "fraud probability, risk tier, and the exact feature values used by the model."
    ),
)
async def score_transaction(
    request: Request,
    body: TransactionRequest,
) -> ScoreResponse:
    """
    Score a single transaction for fraud risk.

    The endpoint:
      1. Validates the request body with Pydantic (400 / 422 on invalid input).
      2. Engineers model features from the raw fields.
      3. Calls FraudPredictor.score() (XGBoost inference, <5ms on CPU).
      4. Returns the probability, risk tier, and engineered features.

    The scored transaction is stored in FraudPredictor's in-memory store and
    can be retrieved later by transaction_id for SHAP explanation via
    POST /explain/{transaction_id}.
    """
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Fraud model is not loaded. The server may still be starting up, "
                "or the model artifacts are missing. Check GET /health/ready."
            ),
        )

    # ── Feature engineering ───────────────────────────────────────────────────
    try:
        txn_features, velocity_estimated = _engineer_features(body)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Feature engineering failed: {exc}",
        ) from exc

    # ── Model inference ────────────────────────────────────────────────────────
    try:
        result = predictor.score(txn_features)
    except ValueError as exc:
        # ValueError means a required feature column was missing or an unknown
        # merchant_category slipped through. This is a caller error.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Scoring failed — invalid input: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scoring failed — internal error: {exc}",
        ) from exc

    # ── Assemble response ─────────────────────────────────────────────────────
    return ScoreResponse(
        transaction_id=result.transaction_id,
        fraud_probability=round(result.fraud_probability, 4),
        is_fraud=result.is_fraud,
        risk_level=result.risk_level,
        scored_at=result.scored_at,
        features_used=FeaturesUsed(
            amount=txn_features["amount"],
            amount_log=txn_features["amount_log"],
            amount_x_risk=txn_features["amount_x_risk"],
            amount_sum_1h=txn_features["amount_sum_1h"],
            txn_count_1h=txn_features["txn_count_1h"],
            hour_of_day=txn_features["hour_of_day"],
            day_of_week=txn_features["day_of_week"],
            is_night=txn_features["is_night"],
            is_weekend=txn_features["is_weekend"],
            account_age_bucket=txn_features["account_age_bucket"],
            merchant_category=txn_features["merchant_category"],
            velocity_estimated=velocity_estimated,
        ),
    )
