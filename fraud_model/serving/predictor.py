"""
Fraud model inference wrapper for the FinSight API.

Loads model artifacts once at process startup and exposes three public methods:

    score(transaction)               → ScoreResult
    batch_score(list[transaction])   → list[ScoreResult]
    get_transaction(transaction_id)  → ScoreResult | None

Scored transactions are kept in an in-memory OrderedDict so the API can later
retrieve them for SHAP explanation without re-running inference.

Usage
-----
    from fraud_model.serving.predictor import FraudPredictor

    predictor = FraudPredictor()           # load once at API startup
    result = predictor.score(transaction)
    print(result.risk_level, result.fraud_probability)
"""

import json
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# ── Artifact paths ─────────────────────────────────────────────────────────────
ARTIFACTS_DIR = Path("fraud_model/artifacts")

# ── Risk level thresholds ──────────────────────────────────────────────────────
# These map raw fraud probability to an actionable risk tier for downstream
# alerting and routing. Thresholds are deliberately asymmetric — fraud costs
# far more than a false positive, so CRITICAL kicks in at 85%.
#
#   LOW      < 30%   — allow transaction, no action
#   MEDIUM  30–60%   — flag for soft review (secondary auth, user notification)
#   HIGH    60–85%   — route to manual review queue
#   CRITICAL ≥ 85%   — block immediately, trigger case creation
RISK_THRESHOLDS = [
    (0.85, "CRITICAL"),
    (0.60, "HIGH"),
    (0.30, "MEDIUM"),
    (0.00, "LOW"),
]

# Cap in-memory store to avoid unbounded growth in long-running processes.
# At ~1 KB per result, 100k entries ≈ 100 MB — a safe ceiling for a single pod.
# When the cap is hit, the oldest 10% of entries are evicted (FIFO order).
MAX_STORE_SIZE = 100_000
EVICT_COUNT    = MAX_STORE_SIZE // 10


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScoreResult:
    """
    Output of a single fraud scoring call.

    Fields
    ------
    transaction_id   : str    UUID of the scored transaction.
    fraud_probability: float  Raw model output in [0, 1].
    risk_level       : str    LOW / MEDIUM / HIGH / CRITICAL.
    is_fraud         : bool   Hard label at the F1-optimal threshold from training.
    scored_at        : str    ISO-8601 UTC timestamp of when scoring ran.
    features         : dict   The encoded feature values used for inference.
                              Stored so the SHAP explainer can be called later
                              without re-sending the transaction payload.
    """
    transaction_id:    str
    fraud_probability: float
    risk_level:        str
    is_fraud:          bool
    scored_at:         str
    features:          dict

    def to_dict(self) -> dict:
        """JSON-serialisable representation (drops the features dict by default
        to keep API responses lean — include explicitly when needed)."""
        return {
            "transaction_id":    self.transaction_id,
            "fraud_probability": round(self.fraud_probability, 4),
            "risk_level":        self.risk_level,
            "is_fraud":          self.is_fraud,
            "scored_at":         self.scored_at,
        }


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────

class FraudPredictor:
    """
    Single-instance inference wrapper. Instantiate once at application startup
    (e.g. in a FastAPI lifespan handler) and reuse across all requests.
    Thread-safe for reads; the in-memory store uses simple dict operations
    which are GIL-protected in CPython.
    """

    def __init__(self, artifacts_dir: Path | str = ARTIFACTS_DIR) -> None:
        artifacts_dir = Path(artifacts_dir)
        self._assert_artifacts_exist(artifacts_dir)

        # Loading XGBoost + SHAP explainer takes ~1–2 s — acceptable at startup,
        # unacceptable if done per request.
        self.model     = joblib.load(artifacts_dir / "fraud_model.joblib")
        self.explainer = joblib.load(artifacts_dir / "shap_explainer.joblib")

        meta = json.loads((artifacts_dir / "feature_cols.json").read_text())
        self.feature_cols      = meta["feature_cols"]
        self.threshold         = meta["best_threshold"]       # F1-optimal from training
        self.category_mapping  = meta["merchant_category_mapping"]  # str → int

        # OrderedDict preserves insertion order, enabling FIFO eviction
        self._store: OrderedDict[str, ScoreResult] = OrderedDict()

        print(
            f"FraudPredictor ready  "
            f"features={len(self.feature_cols)}  "
            f"threshold={self.threshold:.3f}  "
            f"store_cap={MAX_STORE_SIZE:,}"
        )

    # ── score ─────────────────────────────────────────────────────────────────

    def score(self, transaction: dict[str, Any]) -> ScoreResult:
        """
        Score a single transaction and return a fraud probability + risk level.

        Parameters
        ----------
        transaction : dict
            Must contain all keys listed in feature_cols (see feature_cols.json).
            merchant_category may be a raw string ("wire_transfer") — encoding
            is handled internally. An optional "transaction_id" key is used as
            the store key; a UUID is generated if absent.

        Returns
        -------
        ScoreResult
            Stored internally by transaction_id for later retrieval via
            get_transaction(). Calling score() twice with the same id
            overwrites the previous result.
        """
        txn_id  = str(transaction.get("transaction_id", uuid.uuid4()))
        row_df  = self._encode([transaction])

        proba      = float(self.model.predict_proba(row_df)[0, 1])
        risk_level = self._to_risk_level(proba)
        is_fraud   = proba >= self.threshold

        result = ScoreResult(
            transaction_id    = txn_id,
            fraud_probability = proba,
            risk_level        = risk_level,
            is_fraud          = is_fraud,
            scored_at         = datetime.now(timezone.utc).isoformat(),
            features          = row_df.iloc[0].to_dict(),
        )
        self._store_result(txn_id, result)
        return result

    # ── batch_score ───────────────────────────────────────────────────────────

    def batch_score(self, transactions: list[dict[str, Any]]) -> list[ScoreResult]:
        """
        Score a list of transactions in a single vectorised model call.

        Prefer this over calling score() in a loop — XGBoost's predict_proba
        amortises tree traversal across all rows in one C++ call, giving
        roughly 10–20× higher throughput than N individual calls.

        Parameters
        ----------
        transactions : list[dict]
            Each dict follows the same contract as score(). Missing
            transaction_id keys are auto-generated per row.

        Returns
        -------
        list[ScoreResult]
            One result per input transaction, in the same order as the input.
            All results are stored internally for later retrieval.
        """
        if not transactions:
            return []

        txn_ids = [str(t.get("transaction_id", uuid.uuid4())) for t in transactions]
        batch_df = self._encode(transactions)

        # Single predict_proba call across all rows
        probas = self.model.predict_proba(batch_df)[:, 1]

        scored_at = datetime.now(timezone.utc).isoformat()
        results = []
        for i, (txn_id, proba) in enumerate(zip(txn_ids, probas)):
            proba = float(proba)
            result = ScoreResult(
                transaction_id    = txn_id,
                fraud_probability = proba,
                risk_level        = self._to_risk_level(proba),
                is_fraud          = proba >= self.threshold,
                scored_at         = scored_at,
                features          = batch_df.iloc[i].to_dict(),
            )
            self._store_result(txn_id, result)
            results.append(result)

        return results

    # ── get_transaction ───────────────────────────────────────────────────────

    def get_transaction(self, transaction_id: str) -> ScoreResult | None:
        """
        Retrieve a previously scored transaction from the in-memory store.

        Returns None if the transaction_id is unknown (never scored, or evicted
        after the store exceeded MAX_STORE_SIZE). In that case the caller should
        re-submit the transaction for scoring.

        The stored `features` dict on the returned ScoreResult can be passed
        directly to SHAPAnalyzer.explain() for a full breakdown.
        """
        return self._store.get(transaction_id)

    # ── store statistics ──────────────────────────────────────────────────────

    def store_size(self) -> int:
        """Number of transactions currently held in the in-memory store."""
        return len(self._store)

    def risk_summary(self) -> dict[str, int]:
        """
        Count of stored transactions per risk level.
        Useful for health-check endpoints and operational dashboards.
        """
        summary: dict[str, int] = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        for r in self._store.values():
            summary[r.risk_level] += 1
        return summary

    # ── private helpers ───────────────────────────────────────────────────────

    def _encode(self, transactions: list[dict]) -> pd.DataFrame:
        """
        Convert a list of raw transaction dicts into a DataFrame with the
        exact column order and dtypes the model expects.

        Validates that all required feature keys are present and converts
        merchant_category strings to the label-encoded integers used at
        training time.
        """
        rows = []
        for txn in transactions:
            missing = set(self.feature_cols) - set(txn.keys())
            if missing:
                raise ValueError(
                    f"Transaction missing required features: {sorted(missing)}\n"
                    f"Required: {self.feature_cols}"
                )
            row = {k: txn[k] for k in self.feature_cols}

            # Encode merchant_category if passed as a human-readable string.
            # The model was trained on integer codes — passing the raw string
            # would silently produce NaN and degrade prediction quality.
            if isinstance(row["merchant_category"], str):
                cat = row["merchant_category"]
                if cat not in self.category_mapping:
                    raise ValueError(
                        f"Unknown merchant_category '{cat}'. "
                        f"Valid: {sorted(self.category_mapping)}"
                    )
                row["merchant_category"] = self.category_mapping[cat]

            rows.append(row)

        return pd.DataFrame(rows, columns=self.feature_cols)

    def _to_risk_level(self, proba: float) -> str:
        """Map a raw probability to the first matching risk tier."""
        for threshold, level in RISK_THRESHOLDS:
            if proba >= threshold:
                return level
        return "LOW"

    def _store_result(self, txn_id: str, result: ScoreResult) -> None:
        """
        Insert or update a result in the store, evicting the oldest entries
        when MAX_STORE_SIZE is reached. OrderedDict.move_to_end() isn't used
        here — on update we just overwrite in place, keeping the original
        insertion position to preserve approximate FIFO eviction order.
        """
        if txn_id not in self._store and len(self._store) >= MAX_STORE_SIZE:
            # Evict oldest EVICT_COUNT entries to amortise eviction cost
            for _ in range(EVICT_COUNT):
                self._store.popitem(last=False)
        self._store[txn_id] = result

    @staticmethod
    def _assert_artifacts_exist(d: Path) -> None:
        for name in ("fraud_model.joblib", "shap_explainer.joblib", "feature_cols.json"):
            p = d / name
            if not p.exists():
                raise FileNotFoundError(
                    f"Artifact not found: {p}\n"
                    "Run fraud_model/training/train.py first."
                )


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    predictor = FraudPredictor()

    # ── Single transaction ────────────────────────────────────────────────────
    normal = {
        "transaction_id":    "txn-001",
        "amount":            38.50,
        "amount_log":        3.66,
        "amount_x_risk":     1.93,
        "amount_sum_1h":     38.50,
        "txn_count_1h":      1,
        "hour_of_day":       11,
        "day_of_week":       1,
        "is_night":          0,
        "is_weekend":        0,
        "account_age_bucket":2,
        "merchant_category": "grocery",
    }
    r = predictor.score(normal)
    print(f"[score]  {r.transaction_id}  p={r.fraud_probability:.4f}  {r.risk_level}")

    # ── Retrieve from store ───────────────────────────────────────────────────
    stored = predictor.get_transaction("txn-001")
    print(f"[store]  retrieved: {stored.transaction_id}  risk={stored.risk_level}")

    # ── Batch scoring ─────────────────────────────────────────────────────────
    batch = [
        {
            "transaction_id":    f"txn-{i:03d}",
            "amount":            round(200 * (i + 1), 2),
            "amount_log":        round(__import__("math").log1p(200 * (i + 1)), 4),
            "amount_x_risk":     round(200 * (i + 1) * (0.05 if i % 2 == 0 else 0.90), 2),
            "amount_sum_1h":     round(200 * (i + 1) * 3, 2),
            "txn_count_1h":      i + 1,
            "hour_of_day":       2 if i % 3 == 0 else 14,
            "day_of_week":       6 if i % 4 == 0 else 2,
            "is_night":          1 if i % 3 == 0 else 0,
            "is_weekend":        1 if i % 4 == 0 else 0,
            "account_age_bucket":0 if i < 2 else 2,
            "merchant_category": "wire_transfer" if i % 3 == 0 else "retail",
        }
        for i in range(6)
    ]
    results = predictor.batch_score(batch)
    print(f"\n[batch]  {len(results)} transactions scored")
    for r in results:
        print(f"  {r.transaction_id}  p={r.fraud_probability:.4f}  {r.risk_level:<8}  fraud={r.is_fraud}")

    print(f"\n[store]  size={predictor.store_size()}  summary={predictor.risk_summary()}")
