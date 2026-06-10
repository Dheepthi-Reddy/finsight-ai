"""
SageMaker inference handler for the FinSight XGBoost fraud detection model.

What this file is and where it runs
-------------------------------------
This script does NOT run locally. It is packaged inside the model.tar.gz
artifact that deploy.py uploads to S3, and SageMaker extracts it to
/opt/ml/model/code/inference.py inside the inference container at startup.

The SageMaker XGBoost serving container looks for a file named inference.py
(pointed to by the SAGEMAKER_PROGRAM environment variable set in deploy.py).
When a prediction request arrives, the container calls the four handler
functions defined here in this exact sequence:

  1. model_fn(model_dir)
     Called once at container startup. Loads the model and metadata from
     /opt/ml/model/ (where SageMaker extracts the tar.gz contents).
     The return value is passed as the first argument to predict_fn() for
     every subsequent request.

  2. input_fn(request_body, content_type)
     Called for every request. Deserialises the raw HTTP request body into
     a Python object (here, a dict of feature values). The content_type
     argument comes from the HTTP Content-Type header sent by the caller.

  3. predict_fn(input_data, model)
     Called for every request after input_fn(). Receives the deserialised
     input and the model object from model_fn(), runs inference, and returns
     a raw prediction object. No serialisation happens here — that is
     output_fn()'s responsibility.

  4. output_fn(prediction, accept_type)
     Called for every request after predict_fn(). Serialises the raw
     prediction into the format the caller requested via the HTTP Accept
     header.

Why this separation exists
---------------------------
The four-function contract lets SageMaker:
  - Reuse the loaded model across thousands of requests without reloading.
  - Run input and output serialisation on the main serving thread while
    offloading predict_fn() to a thread pool for concurrency.
  - Apply health checks and canary traffic splitting at the HTTP layer
    without touching the inference logic.

This file vs predictor.py
--------------------------
fraud_model/serving/predictor.py is the in-process inference wrapper used by
the FastAPI server. It loads artifacts using relative paths and returns rich
ScoreResult dataclasses.

This file is the SageMaker-specific inference handler. It:
  - Loads artifacts from /opt/ml/model/ (the SageMaker extraction path).
  - Accepts raw JSON feature dicts rather than full transaction payloads.
  - Returns a minimal JSON response (probability + label + risk level) rather
    than a full ScoreResult, to keep the network payload small.
  - Has no FastAPI dependency — it is pure Python with only joblib, pandas,
    numpy, and xgboost, all of which are pre-installed in the SageMaker
    XGBoost container.

Contract
--------
Request body (application/json):
    {
        "amount":           150.0,
        "amount_log":       5.01,
        "amount_x_risk":    225.0,
        "amount_sum_1h":    450.0,
        "txn_count_1h":     3,
        "hour_of_day":      14,
        "day_of_week":      2,
        "is_night":         0,
        "is_weekend":       0,
        "account_age_bucket": 2,
        "merchant_category": 7
    }
    Note: merchant_category must be the integer code from feature_cols.json,
    not the raw string. Encode it on the caller side using the category_mapping.

Response body (application/json):
    {
        "fraud_probability": 0.0312,
        "is_fraud":          false,
        "risk_level":        "LOW"
    }
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

# ── Risk level thresholds ──────────────────────────────────────────────────────
# Must be kept in sync with fraud_model/serving/predictor.py.
# Duplicated here so this file has no import dependency on the rest of the
# FinSight package — SageMaker containers don't have the project installed.
_RISK_THRESHOLDS = [
    (0.85, "CRITICAL"),
    (0.60, "HIGH"),
    (0.30, "MEDIUM"),
    (0.00, "LOW"),
]


# ─────────────────────────────────────────────────────────────────────────────
# HANDLER FUNCTION 1 — model loading (called once at container startup)
# ─────────────────────────────────────────────────────────────────────────────

def model_fn(model_dir: str) -> dict[str, Any]:
    """
    Load model artifacts from the SageMaker model directory.

    Called once when the container starts. The return value (here, a dict
    bundling the model and metadata) is passed as the second argument to
    every predict_fn() call. Loading once and reusing is critical for latency:
    XGBoost deserialization takes ~200ms; amortised over thousands of requests
    it is negligible, but per-request it would dominate the response time.

    Parameters
    ----------
    model_dir : str
        Path SageMaker extracts the model.tar.gz to. Always /opt/ml/model/
        on real SageMaker instances; can be overridden for local testing by
        setting the SM_MODEL_DIR environment variable.

    Returns
    -------
    dict with keys:
        model            : XGBoost sklearn wrapper (predict_proba available)
        feature_cols     : list[str]  ordered feature names
        threshold        : float      F1-optimal classification threshold
        category_mapping : dict       raw category string → int code
    """
    model_dir = Path(model_dir)

    model_path     = model_dir / "fraud_model.joblib"
    meta_path      = model_dir / "feature_cols.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"fraud_model.joblib not found in {model_dir}. "
            "Ensure the model artifact was packaged correctly by deploy.py."
        )
    if not meta_path.exists():
        raise FileNotFoundError(
            f"feature_cols.json not found in {model_dir}. "
            "Ensure the metadata was packaged with the model artifact."
        )

    model = joblib.load(model_path)
    meta  = json.loads(meta_path.read_text())

    print(
        f"[model_fn] Loaded model from {model_path}  "
        f"features={len(meta['feature_cols'])}  "
        f"threshold={meta['best_threshold']:.3f}"
    )

    return {
        "model":            model,
        "feature_cols":     meta["feature_cols"],
        "threshold":        meta["best_threshold"],
        "category_mapping": meta.get("merchant_category_mapping", {}),
    }


# ─────────────────────────────────────────────────────────────────────────────
# HANDLER FUNCTION 2 — input deserialisation
# ─────────────────────────────────────────────────────────────────────────────

def input_fn(request_body: str | bytes, content_type: str) -> dict[str, Any]:
    """
    Deserialise the raw HTTP request body into a feature dict.

    SageMaker calls this before predict_fn(). The content_type argument
    mirrors the HTTP Content-Type header sent by the caller. We support only
    application/json — raising ValueError for any other type causes SageMaker
    to return HTTP 400 to the caller with the error message in the body.

    Parameters
    ----------
    request_body : str | bytes   Raw HTTP body bytes from the POST request.
    content_type : str           HTTP Content-Type header value.

    Returns
    -------
    dict[str, Any]   Feature name → value mapping ready for predict_fn().
    """
    if content_type != "application/json":
        raise ValueError(
            f"Unsupported content type: {content_type!r}. "
            "Send requests with Content-Type: application/json."
        )

    if isinstance(request_body, bytes):
        request_body = request_body.decode("utf-8")

    features = json.loads(request_body)

    if not isinstance(features, dict):
        raise ValueError(
            f"Expected a JSON object (dict) but received {type(features).__name__}. "
            "Send a flat dict of feature_name → value."
        )

    return features


# ─────────────────────────────────────────────────────────────────────────────
# HANDLER FUNCTION 3 — inference
# ─────────────────────────────────────────────────────────────────────────────

def predict_fn(
    input_data: dict[str, Any],
    model_assets: dict[str, Any],
) -> dict[str, Any]:
    """
    Run fraud inference on the deserialised feature dict.

    Constructs a single-row DataFrame in the exact column order the model was
    trained on (preserved in feature_cols.json), calls predict_proba(), and
    returns a dict of raw prediction values. No serialisation here — that is
    output_fn()'s responsibility.

    Parameters
    ----------
    input_data    : dict   Feature name → value from input_fn().
    model_assets  : dict   The object returned by model_fn().

    Returns
    -------
    dict with keys: fraud_probability (float), is_fraud (bool), risk_level (str).
    """
    model        = model_assets["model"]
    feature_cols = model_assets["feature_cols"]
    threshold    = model_assets["threshold"]

    # Validate that all required features are present. Missing features would
    # produce NaN values in the DataFrame, silently degrading model accuracy
    # instead of raising an obvious error.
    missing = [col for col in feature_cols if col not in input_data]
    if missing:
        raise ValueError(
            f"Missing required features: {missing}. "
            f"Expected all of: {feature_cols}"
        )

    # Build a single-row DataFrame in the model's training column order.
    # Column order matters for XGBoost — it uses positional feature indices
    # internally, so swapping two columns would silently corrupt predictions.
    row = {col: [input_data[col]] for col in feature_cols}
    df  = pd.DataFrame(row)

    # predict_proba returns shape (1, 2): [[P(legit), P(fraud)]].
    # We take the fraud probability (column index 1).
    fraud_probability = float(model.predict_proba(df)[0, 1])
    is_fraud          = fraud_probability >= threshold
    risk_level        = _to_risk_level(fraud_probability)

    return {
        "fraud_probability": fraud_probability,
        "is_fraud":          is_fraud,
        "risk_level":        risk_level,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HANDLER FUNCTION 4 — output serialisation
# ─────────────────────────────────────────────────────────────────────────────

def output_fn(prediction: dict[str, Any], accept_type: str) -> tuple[str, str]:
    """
    Serialise the prediction dict into the HTTP response body.

    SageMaker calls this after predict_fn() with the HTTP Accept header value
    as accept_type. Returning a (body, content_type) tuple lets SageMaker set
    the correct Content-Type header on the response.

    We only support application/json. For CSV output (e.g. for SageMaker Batch
    Transform), add an "text/csv" branch here.

    Parameters
    ----------
    prediction  : dict    The dict returned by predict_fn().
    accept_type : str     HTTP Accept header value from the caller.

    Returns
    -------
    tuple[str, str]   (response_body_string, content_type_header_value)
    """
    if accept_type not in ("application/json", "*/*", ""):
        raise ValueError(
            f"Unsupported accept type: {accept_type!r}. "
            "Set Accept: application/json in your request."
        )

    # Round the probability to 4 decimal places for a cleaner response payload.
    output = {
        "fraud_probability": round(prediction["fraud_probability"], 4),
        "is_fraud":          prediction["is_fraud"],
        "risk_level":        prediction["risk_level"],
    }

    return json.dumps(output), "application/json"


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _to_risk_level(probability: float) -> str:
    """
    Map a fraud probability to a human-readable risk tier.

    Thresholds mirror fraud_model/serving/predictor.py exactly.
    Duplicated to keep this file self-contained inside the SageMaker container.
    """
    for threshold, level in _RISK_THRESHOLDS:
        if probability >= threshold:
            return level
    return "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL TEST HARNESS
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate the SageMaker container invoking the handler functions locally.
    # Set SM_MODEL_DIR to the local artifacts directory for testing.
    model_dir = os.environ.get("SM_MODEL_DIR", "fraud_model/artifacts")

    print(f"Loading model from {model_dir}...")
    assets = model_fn(model_dir)
    print(f"  feature_cols : {assets['feature_cols']}")
    print(f"  threshold    : {assets['threshold']:.3f}")

    # Sample feature dict — all numeric, merchant_category as integer code.
    # These values are not meaningful; they exercise the inference path only.
    sample_features = {
        "amount":             150.00,
        "amount_log":         5.0106,
        "amount_x_risk":      225.00,
        "amount_sum_1h":      450.00,
        "txn_count_1h":       3,
        "hour_of_day":        14,
        "day_of_week":        2,
        "is_night":           0,
        "is_weekend":         0,
        "account_age_bucket": 2,
        "merchant_category":  7,
    }

    print("\nRunning local inference test...")
    raw_input  = input_fn(json.dumps(sample_features), "application/json")
    prediction = predict_fn(raw_input, assets)
    body, ct   = output_fn(prediction, "application/json")

    print(f"  Response ({ct}): {body}")
    print("\nAll four handler functions executed successfully.")
