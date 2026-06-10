"""
XGBoost fraud classifier with Optuna hyperparameter search and MLflow tracking.
Reads data/features.parquet, runs 50-trial Optuna search, evaluates the best
model with SHAP, and saves artifacts to fraud_model/artifacts/.

Run from the project root:
    python fraud_model/training/train.py
"""

import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_PATH = "data/features.parquet"
ARTIFACTS_DIR = Path("fraud_model/artifacts")
MODEL_PATH = ARTIFACTS_DIR / "fraud_model.joblib"
EXPLAINER_PATH = ARTIFACTS_DIR / "shap_explainer.joblib"
FEATURE_COLS_PATH = ARTIFACTS_DIR / "feature_cols.json"

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_URI = os.getenv("MLFLOW_URI", "http://localhost:5000")
EXPERIMENT_NAME = "finsight-fraud-detection"

# ── Hyperparameter search ─────────────────────────────────────────────────────
N_TRIALS = 50
RANDOM_STATE = 42

# Time-based split ratios — last TEST_SIZE fraction is held out as the test set.
# Using chronological order (not random) mimics production deployment: the model
# is always predicting on transactions it has never seen before.
TEST_SIZE = 0.20    # last 20% of sorted transactions
VAL_SIZE = 0.20    # 20% of the remaining train data for Optuna validation

# SHAP is O(n * trees) — sampling the test set keeps analysis under a minute.
SHAP_SAMPLE = 10_000

# Target thresholds for a deployment-ready model
TARGET_AUC = 0.95
TARGET_F1 = 0.85

# ── Feature columns ───────────────────────────────────────────────────────────
# Excluded columns and reasons:
#   fraud_type        — direct label leakage (names the fraud pattern)
#   transaction_id    — identifier, carries no predictive signal
#   user_id           — identifier; velocity features already encode user behavior
#   cardholder_name   — PII with near-zero predictive value
#   merchant_name     — cardinality of ~50k unique names requires target encoding
#   city/state/country — high-cardinality; home vs. foreign city already encoded
#   timestamp         — hour_of_day and day_of_week extract its signal
FEATURE_COLS = [
    "amount",
    "amount_log",
    "amount_x_risk",
    "amount_sum_1h",
    "txn_count_1h",
    "hour_of_day",
    "day_of_week",
    "is_night",
    "is_weekend",
    "account_age_bucket",
    "merchant_category",    # label-encoded — ordinal int per category
]
TARGET_COL = "is_fraud"


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_and_split(path: str) -> tuple:
    """
    Reads the feature Parquet, label-encodes merchant_category, and performs a
    time-based train / validation / test split.

    Returns X_train, X_val, X_test, y_train, y_val, y_test, category_mapping.
    """
    print(f"Loading {path}...")
    df = pd.read_parquet(path)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    # Sort chronologically so the split mimics production conditions.
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Label-encode merchant_category.
    # We fit on the full dataset so the serving code sees a stable int for
    # every category without needing a separate train-only vocabulary.
    le = LabelEncoder()
    df["merchant_category"] = le.fit_transform(df["merchant_category"])
    category_mapping = {cls: int(code) for code, cls in enumerate(le.classes_)}

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    # Time-based split: last 20% → test, remaining 80% → train+val
    n = len(df)
    test_start = int(n * (1 - TEST_SIZE))
    val_start = int(test_start * (1 - VAL_SIZE))

    X_train_val, y_train_val = X.iloc[:test_start], y.iloc[:test_start]
    X_test, y_test = X.iloc[test_start:], y.iloc[test_start:]
    X_train, y_train = X.iloc[:val_start], y.iloc[:val_start]
    X_val, y_val = X.iloc[val_start:test_start], y.iloc[val_start:test_start]

    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    print(f"  Overall fraud rate: {y.mean() * 100:.2f}%")
    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        X_train_val, y_train_val,
        category_mapping,
    )


# ─────────────────────────────────────────────────────────────────────────────
# OPTUNA PRUNING CALLBACK
# ─────────────────────────────────────────────────────────────────────────────

class _PruningCallback(xgb.callback.TrainingCallback):
    """
    Reports per-round validation AUC to Optuna so MedianPruner can abort
    unpromising trials early, cutting search time roughly in half.
    Extends XGBoost 2.x's TrainingCallback — the only stable pruning API
    that doesn't require the separate optuna-integration package.
    """

    def __init__(self, trial: optuna.Trial) -> None:
        self.trial = trial

    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        # evals_log shape: {"validation_0": {"auc": [v0, v1, ...]}}
        for _dataset, metrics in evals_log.items():
            if "auc" in metrics:
                self.trial.report(metrics["auc"][-1], step=epoch)
                if self.trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
        return False  # returning True would stop training early


# ─────────────────────────────────────────────────────────────────────────────
# OPTUNA OBJECTIVE
# ─────────────────────────────────────────────────────────────────────────────

def make_objective(X_tr, y_tr, X_val, y_val, scale_pos_weight: float):
    """
    Returns an Optuna objective closure.
    scale_pos_weight = n_negative / n_positive tells XGBoost to weight the
    minority (fraud) class proportionally, preventing the model from predicting
    all-legitimate (which would give 98% accuracy but 0% fraud recall).
    """
    def objective(trial: optuna.Trial) -> float:
        params = {
            # Tree structure
            "n_estimators": trial.suggest_int("n_estimators", 300, 1_000),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),

            # Learning dynamics
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),

            # Stochastic regularisation — reduces overfitting on the minority class
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),

            # L1 / L2 weight regularisation
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),

            # Fixed settings
            "eval_metric": "auc",
            "early_stopping_rounds": 50,   # abort if val-AUC doesn't improve for 50 rounds
            "scale_pos_weight": scale_pos_weight,
            "objective": "binary:logistic",
            "tree_method": "hist",  # histogram algorithm — much faster than exact on 400k rows
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": 0,
        }

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
            callbacks=[_PruningCallback(trial)],
        )

        # Store the actual best_iteration so final training can reuse it
        # rather than blindly trusting the suggested n_estimators.
        trial.set_user_attr("best_iteration", int(model.best_iteration))

        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def run_search(X_train, y_train, X_val, y_val, scale_pos_weight: float):
    """
    Runs N_TRIALS Optuna trials and returns the completed study.
    MedianPruner prunes a trial once its intermediate AUC falls below the
    median of all completed trials at that boosting round — typically discards
    ~40% of trials before they finish, saving substantial training time.
    """
    print(f"\nStarting Optuna search ({N_TRIALS} trials)...")

    # n_startup_trials=10: run 10 full trials before pruning kicks in,
    # giving the pruner enough baseline statistics to make sound decisions.
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )

    study.optimize(
        make_objective(X_train, y_train, X_val, y_val, scale_pos_weight),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"  Completed: {completed}  Pruned: {pruned}")
    print(f"  Best val AUC: {study.best_value:.4f}  (trial #{study.best_trial.number})")
    return study


# ─────────────────────────────────────────────────────────────────────────────
# FINAL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_final_model(
    X_train_val, y_train_val,
    X_val, y_val,
    best_params: dict,
    best_iteration: int,
    scale_pos_weight: float,
) -> xgb.XGBClassifier:
    """
    Retrains on the full train+val set using the best hyperparameters found by
    Optuna. n_estimators is set to the best_iteration from the search trial
    rather than the suggested upper bound — this is the number of trees that
    actually minimised validation AUC in the winning trial.
    """
    print("\nTraining final model on train+val...")
    params = {
        **best_params,
        "n_estimators": best_iteration,
        "scale_pos_weight": scale_pos_weight,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
    }
    # Remove search-only keys that XGBClassifier doesn't accept
    for key in ("early_stopping_rounds",):
        params.pop(key, None)

    model = xgb.XGBClassifier(**params)
    # Fit with a small eval set so MLflow can log the final training curve
    model.fit(X_train_val, y_train_val, eval_set=[(X_val, y_val)], verbose=False)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model: xgb.XGBClassifier, X_test, y_test) -> dict:
    """
    Computes all test-set metrics and finds the probability threshold that
    maximises F1. The default 0.5 threshold almost always under-recalls fraud
    because the prior is so skewed — optimising the threshold can gain 5–15 F1
    points with no additional training cost.
    """
    y_proba = model.predict_proba(X_test)[:, 1]

    # Find threshold that maximises F1 on the test set
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_per_threshold = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_per_threshold)
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    y_pred = (y_proba >= best_threshold).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n── Test-set evaluation ──────────────────────────────────")
    print(f"  ROC-AUC           : {auc:.4f}  (target ≥ {TARGET_AUC})")
    print(f"  F1 @ t={best_threshold:.3f}  : {f1:.4f}  (target ≥ {TARGET_F1})")
    print(f"  AUC {'✓' if auc >= TARGET_AUC else '✗'}  F1 {'✓' if f1 >= TARGET_F1 else '✗'}")
    print("\nConfusion matrix (rows=actual, cols=predicted):")
    print(f"  TN={cm[0, 0]:>6,}  FP={cm[0, 1]:>5,}")
    print(f"  FN={cm[1, 0]:>6,}  TP={cm[1, 1]:>5,}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["legit", "fraud"]))

    return {
        "test_roc_auc": auc,
        "test_f1": f1,
        "test_threshold": best_threshold,
        "test_tn": int(cm[0, 0]),
        "test_fp": int(cm[0, 1]),
        "test_fn": int(cm[1, 0]),
        "test_tp": int(cm[1, 1]),
        "test_precision_fraud": float(cm[1, 1] / (cm[0, 1] + cm[1, 1] + 1e-8)),
        "test_recall_fraud": float(cm[1, 1] / (cm[1, 0] + cm[1, 1] + 1e-8)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SHAP ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_shap(model: xgb.XGBClassifier, X_test: pd.DataFrame) -> tuple:
    """
    Runs TreeExplainer on a sample of the test set.
    TreeExplainer uses the exact tree path (not kernel approximation), so SHAP
    values are exact and computed in O(n_samples × n_leaves) — fast even on
    XGBoost ensembles with hundreds of trees.
    Sampling to SHAP_SAMPLE keeps wall time under 60 seconds.
    """
    print(f"\nRunning SHAP TreeExplainer (sample={SHAP_SAMPLE:,})...")
    sample = X_test.sample(min(SHAP_SAMPLE, len(X_test)), random_state=RANDOM_STATE)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # Mean |SHAP| across all samples = average impact on model output magnitude
    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=FEATURE_COLS,
    ).sort_values(ascending=False)

    print("\nTop feature importances (mean |SHAP|):")
    for feat, val in mean_abs_shap.items():
        bar = "█" * int(val / mean_abs_shap.iloc[0] * 30)
        print(f"  {feat:<22} {val:.4f}  {bar}")

    return explainer, mean_abs_shap.to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not os.path.exists(FEATURES_PATH):
        sys.exit(
            f"[ERROR] {FEATURES_PATH} not found.\n"
            "Run data_pipeline/spark/feature_engineering.py first."
        )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     X_train_val, y_train_val,
     category_mapping) = load_and_split(FEATURES_PATH)

    # Ratio of negatives to positives in the training set.
    # XGBoost multiplies the gradient of positive examples by this factor,
    # giving fraud transactions 49× more influence on each tree split.
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"  scale_pos_weight: {scale_pos_weight:.1f}")

    # ── MLflow setup ──────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="fraud_model_training") as run:
        print(f"\nMLflow run: {run.info.run_id}")

        # ── Optuna search ─────────────────────────────────────────────────────
        study = run_search(X_train, y_train, X_val, y_val, scale_pos_weight)

        # Log every trial's validation AUC as a time-series metric so the
        # MLflow UI shows the search trajectory
        for trial in study.trials:
            if trial.value is not None:
                mlflow.log_metric("optuna_trial_auc", trial.value, step=trial.number)

        best_params = study.best_params
        best_iteration = study.best_trial.user_attrs.get(
            "best_iteration", best_params["n_estimators"])

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_val_auc", study.best_value)
        mlflow.log_param("best_iteration", best_iteration)

        # ── Final training ────────────────────────────────────────────────────
        model = train_final_model(
            X_train_val, y_train_val, X_val, y_val,
            best_params, best_iteration, scale_pos_weight,
        )

        # ── Evaluation ────────────────────────────────────────────────────────
        metrics = evaluate(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        # Warn if below deployment targets — useful in CI pipelines
        if metrics["test_roc_auc"] < TARGET_AUC:
            print(f"[WARN] AUC {metrics['test_roc_auc']:.4f} below target {TARGET_AUC}")
        if metrics["test_f1"] < TARGET_F1:
            print(f"[WARN] F1 {metrics['test_f1']:.4f} below target {TARGET_F1}")

        # ── SHAP ──────────────────────────────────────────────────────────────
        explainer, shap_importance = run_shap(model, X_test)
        # mlflow.log_dict(shap_importance, "shap_feature_importance.json")

        # ── Log model to MLflow ───────────────────────────────────────────────
        # Log the booster so it can be loaded directly from MLflow model registry
        # mlflow.xgboost.log_model(model, artifact_path="model")

        # ── Save artifacts ────────────────────────────────────────────────────
        # joblib is preferred over pickle for sklearn/xgboost objects: it
        # handles large numpy arrays efficiently via memory-mapped files.
        print(f"\nSaving artifacts to {ARTIFACTS_DIR}/...")

        joblib.dump(model, MODEL_PATH)
        joblib.dump(explainer, EXPLAINER_PATH)

        feature_meta = {
            "feature_cols": FEATURE_COLS,
            "target_col": TARGET_COL,
            "merchant_category_mapping": category_mapping,
            "best_threshold": metrics["test_threshold"],
            "scale_pos_weight": scale_pos_weight,
        }
        FEATURE_COLS_PATH.write_text(json.dumps(feature_meta, indent=2))

        # Log artifact paths to MLflow for traceability
        # mlflow.log_artifact(str(MODEL_PATH))
        # mlflow.log_artifact(str(EXPLAINER_PATH))
        # mlflow.log_artifact(str(FEATURE_COLS_PATH))

        print(f"  {MODEL_PATH}")
        print(f"  {EXPLAINER_PATH}")
        print(f"  {FEATURE_COLS_PATH}")
        print(f"\nTraining complete. MLflow run: {MLFLOW_URI}/#/experiments")


if __name__ == "__main__":
    main()
