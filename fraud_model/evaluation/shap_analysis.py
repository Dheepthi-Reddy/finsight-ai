"""
SHAP explainability for the FinSight fraud detection model.

How to read SHAP values
-----------------------
TreeExplainer computes exact Shapley values for tree ensembles. Each value
answers: "How much did this feature push the prediction away from the base
rate — toward fraud or toward legitimate?"

  base_value   — the model's average log-odds prediction across all training
                 data (reflects the ~2% fraud rate; converts to ~0.02 proba).
  shap_value   — per-feature contribution, also in log-odds. Positive = pushed
                 toward fraud; negative = pushed toward legitimate.
  prediction   — sigmoid(base_value + Σ shap_values) ≈ predict_proba output.

Example: base = -3.9 (≈2% fraud), amount_x_risk SHAP = +2.1, all others = 0
  → sigmoid(-3.9 + 2.1) = sigmoid(-1.8) ≈ 14% fraud probability.

Usage
-----
    from fraud_model.evaluation.shap_analysis import SHAPAnalyzer

    analyzer = SHAPAnalyzer()
    result = analyzer.explain({
        "amount": 4800.0, "merchant_category": "wire_transfer", ...
    })
    print(result.top_5_drivers)

    analyzer.beeswarm()          # saves plot to fraud_model/artifacts/
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

# matplotlib must be imported after shap to avoid backend conflicts on headless
# servers; "Agg" renders to PNG without needing a display.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Artifact paths (relative to project root) ─────────────────────────────────
ARTIFACTS_DIR     = Path("fraud_model/artifacts")
MODEL_PATH        = ARTIFACTS_DIR / "fraud_model.joblib"
EXPLAINER_PATH    = ARTIFACTS_DIR / "shap_explainer.joblib"
FEATURE_COLS_PATH = ARTIFACTS_DIR / "feature_cols.json"
BEESWARM_PATH     = ARTIFACTS_DIR / "shap_beeswarm.png"

FEATURES_PARQUET = "data/features.parquet"
BEESWARM_SAMPLE  = 5_000   # enough rows to show the full distribution without crowding


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TransactionExplanation:
    """
    Complete explanation for one transaction prediction.

    Fields
    ------
    prediction_proba : float
        Model's estimated probability that this transaction is fraudulent.
        Use `best_threshold` from feature_cols.json to binarise for alerts.
    predicted_label : int
        1 = fraud, 0 = legitimate, at the F1-optimal threshold from training.
    base_value : float
        Expected model output (log-odds) across the training distribution.
        sigmoid(base_value) ≈ dataset fraud rate (~0.02).
    shap_values : dict[str, float]
        Per-feature SHAP contributions in log-odds. Positive = drives toward
        fraud; negative = drives toward legitimate.
    top_5_drivers : list[dict]
        Top 5 features sorted by |SHAP|, each with keys:
          feature, shap_value, feature_value, direction.
    raw_features : dict
        Original (un-encoded) feature values passed into explain().
    """
    prediction_proba:  float
    predicted_label:   int
    base_value:        float
    shap_values:       dict = field(default_factory=dict)
    top_5_drivers:     list = field(default_factory=list)
    raw_features:      dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Fraud probability : {self.prediction_proba:.1%}  "
            f"({'FRAUD' if self.predicted_label else 'LEGITIMATE'})",
            f"Base rate (log-odds base_value → proba): "
            f"{self.base_value:.3f} → {_sigmoid(self.base_value):.3%}",
            "",
            "Top drivers (+ increases fraud risk, − decreases it):",
        ]
        for d in self.top_5_drivers:
            sign = "+" if d["shap_value"] > 0 else ""
            lines.append(
                f"  {d['feature']:<22}  "
                f"value={d['feature_value']!r:<12}  "
                f"SHAP={sign}{d['shap_value']:.4f}"
            )
        return "\n".join(lines)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


# ─────────────────────────────────────────────────────────────────────────────
# ANALYZER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class SHAPAnalyzer:
    """
    Loads saved artifacts once and exposes explain() and beeswarm() methods.
    Designed to be instantiated once at API startup and reused across requests.
    """

    def __init__(self, artifacts_dir: Path | str = ARTIFACTS_DIR) -> None:
        artifacts_dir = Path(artifacts_dir)
        self._assert_artifacts_exist(artifacts_dir)

        self.model     = joblib.load(artifacts_dir / "fraud_model.joblib")
        self.explainer = joblib.load(artifacts_dir / "shap_explainer.joblib")

        meta = json.loads((artifacts_dir / "feature_cols.json").read_text())
        self.feature_cols         = meta["feature_cols"]
        self.threshold            = meta["best_threshold"]
        self.category_mapping     = meta["merchant_category_mapping"]  # str → int

        # TreeExplainer.expected_value is the log-odds base value.
        # For binary XGBoost it may be a 1-element array or a scalar.
        ev = self.explainer.expected_value
        self.base_value = float(ev[1] if hasattr(ev, "__len__") and len(ev) == 2 else ev)

        print(
            f"SHAPAnalyzer loaded  "
            f"features={len(self.feature_cols)}  "
            f"base_value={self.base_value:.4f}  "
            f"(≈{_sigmoid(self.base_value):.2%} fraud base rate)"
        )

    # ── Public: explain one transaction ──────────────────────────────────────

    def explain(self, transaction: dict[str, Any]) -> TransactionExplanation:
        """
        Compute SHAP values for a single transaction.

        Parameters
        ----------
        transaction : dict
            Must contain all keys in feature_cols. merchant_category can be
            passed as a raw string (e.g. "wire_transfer") — encoding is handled
            internally so callers don't need to know the training label mapping.

        Returns
        -------
        TransactionExplanation
            Full breakdown including per-feature SHAP values and top-5 drivers.
        """
        row_df = self._encode_transaction(transaction)

        # shap_values shape: (1, n_features) for the positive (fraud) class
        shap_vals = self.explainer.shap_values(row_df)
        shap_arr  = np.array(shap_vals)[0] if np.array(shap_vals).ndim == 2 else np.array(shap_vals)

        proba         = float(self.model.predict_proba(row_df)[0, 1])
        predicted     = int(proba >= self.threshold)
        shap_by_feat  = {f: float(v) for f, v in zip(self.feature_cols, shap_arr)}

        # Top 5 by absolute SHAP — the features that moved this prediction the most
        sorted_feats = sorted(shap_by_feat.items(), key=lambda kv: abs(kv[1]), reverse=True)
        top_5 = [
            {
                "feature":       feat,
                "shap_value":    val,
                # Show the original (pre-encoded) value when available
                "feature_value": transaction.get(feat, row_df[feat].iloc[0]),
                "direction":     "increases_fraud_risk" if val > 0 else "decreases_fraud_risk",
            }
            for feat, val in sorted_feats[:5]
        ]

        return TransactionExplanation(
            prediction_proba = proba,
            predicted_label  = predicted,
            base_value       = self.base_value,
            shap_values      = shap_by_feat,
            top_5_drivers    = top_5,
            raw_features     = transaction,
        )

    # ── Public: beeswarm summary plot ────────────────────────────────────────

    def beeswarm(
        self,
        df: pd.DataFrame | None = None,
        output_path: Path | str = BEESWARM_PATH,
        n_samples: int = BEESWARM_SAMPLE,
    ) -> Path:
        """
        Generate and save a SHAP beeswarm summary plot.

        The beeswarm plot shows the full distribution of SHAP values across
        many transactions at once:
          • Y-axis: features ranked by mean |SHAP| (most impactful at top)
          • X-axis: SHAP value — right = pushed toward fraud, left = away from it
          • Color:  feature value — red = high, blue = low
          • Each dot: one transaction

        Reading example: if amount_x_risk has a cluster of red dots (high values)
        far to the right, it means high-risk merchant × large amounts strongly
        drive fraud predictions. Blue dots near zero mean low amounts at safe
        merchants have little effect.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Must contain feature_cols + merchant_category (raw or encoded).
            If None, loads from data/features.parquet automatically.
        output_path : Path
            Where to write the PNG. Defaults to fraud_model/artifacts/shap_beeswarm.png
        n_samples : int
            Number of rows to plot. 5,000 is enough to show dense patterns
            without overcrowding the dots.
        """
        if df is None:
            df = self._load_features_parquet()

        # Encode merchant_category if it came in as strings
        if df["merchant_category"].dtype == object:
            df = df.copy()
            df["merchant_category"] = df["merchant_category"].map(self.category_mapping)

        sample = df[self.feature_cols].dropna().sample(
            min(n_samples, len(df)), random_state=42
        )

        print(f"Computing SHAP values for {len(sample):,} transactions...")
        shap_vals = self.explainer.shap_values(sample)

        # shap.summary_plot handles its own figure internally; plt.gcf() captures it.
        shap.summary_plot(
            shap_vals,
            sample,
            plot_type="dot",       # beeswarm (vs "bar" which shows mean |SHAP|)
            feature_names=self.feature_cols,
            max_display=len(self.feature_cols),
            show=False,
        )

        fig = plt.gcf()
        fig.set_size_inches(10, 7)
        plt.title(
            "SHAP Beeswarm — Feature Impact on Fraud Probability\n"
            "Red = high feature value   Blue = low   Right = toward fraud",
            fontsize=11,
            pad=14,
        )
        plt.tight_layout()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Beeswarm plot saved to {output_path}")
        return output_path

    # ── Private helpers ───────────────────────────────────────────────────────

    def _encode_transaction(self, transaction: dict) -> pd.DataFrame:
        """
        Validates input keys, encodes merchant_category, and returns a
        single-row DataFrame in the exact column order the model expects.
        """
        missing = set(self.feature_cols) - set(transaction.keys())
        if missing:
            raise ValueError(f"Transaction missing required features: {missing}")

        row = dict(transaction)

        # Accept merchant_category as a human-readable string or a pre-encoded int
        if isinstance(row["merchant_category"], str):
            cat = row["merchant_category"]
            if cat not in self.category_mapping:
                raise ValueError(
                    f"Unknown merchant_category '{cat}'. "
                    f"Valid values: {sorted(self.category_mapping.keys())}"
                )
            row["merchant_category"] = self.category_mapping[cat]

        return pd.DataFrame([row])[self.feature_cols]

    def _load_features_parquet(self) -> pd.DataFrame:
        if not Path(FEATURES_PARQUET).exists():
            raise FileNotFoundError(
                f"{FEATURES_PARQUET} not found. "
                "Run data_pipeline/spark/feature_engineering.py first, "
                "or pass a DataFrame directly to beeswarm()."
            )
        print(f"Loading {FEATURES_PARQUET} for beeswarm plot...")
        return pd.read_parquet(FEATURES_PARQUET)

    @staticmethod
    def _assert_artifacts_exist(d: Path) -> None:
        for p in (d / "fraud_model.joblib", d / "shap_explainer.joblib", d / "feature_cols.json"):
            if not p.exists():
                raise FileNotFoundError(
                    f"Artifact not found: {p}\n"
                    "Run fraud_model/training/train.py first."
                )


# ─────────────────────────────────────────────────────────────────────────────
# DEMO / CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    analyzer = SHAPAnalyzer()

    # ── Example 1: normal grocery purchase ───────────────────────────────────
    normal_txn = {
        "amount":            42.50,
        "amount_log":        3.77,
        "amount_x_risk":     2.13,    # 42.50 × 0.05 (grocery risk)
        "amount_sum_1h":     42.50,
        "txn_count_1h":      1,
        "hour_of_day":       14,
        "day_of_week":       2,       # Wednesday
        "is_night":          0,
        "is_weekend":        0,
        "account_age_bucket":2,       # established account
        "merchant_category": "grocery",
    }
    print("\n── Example 1: Normal grocery transaction ─────────────────")
    r1 = analyzer.explain(normal_txn)
    print(r1.summary())

    # ── Example 2: high-risk wire transfer at 2 AM ────────────────────────────
    fraud_txn = {
        "amount":            3200.00,
        "amount_log":        8.07,
        "amount_x_risk":     3040.00,  # 3200 × 0.95 (wire_transfer risk)
        "amount_sum_1h":     9800.00,  # large cumulative spend this hour
        "txn_count_1h":      6,        # velocity burst signal
        "hour_of_day":       2,
        "day_of_week":       6,        # Sunday
        "is_night":          1,
        "is_weekend":        1,
        "account_age_bucket":0,        # new account
        "merchant_category": "wire_transfer",
    }
    print("\n── Example 2: Suspicious wire transfer ───────────────────")
    r2 = analyzer.explain(fraud_txn)
    print(r2.summary())

    # ── Generate beeswarm plot ────────────────────────────────────────────────
    print("\n── Beeswarm summary plot ─────────────────────────────────")
    analyzer.beeswarm()
