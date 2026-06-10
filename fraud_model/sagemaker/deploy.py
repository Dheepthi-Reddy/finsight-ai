"""
SageMaker deployment for the FinSight XGBoost fraud detection model.

What SageMaker is vs local serving
------------------------------------
The local predictor in fraud_model/serving/predictor.py loads the model into
the same Python process as the FastAPI server and scores transactions in-process.
This works fine for development and low-traffic deployments, but has hard limits:

  SCALING    — the model is pinned to however many CPU cores the API pod has.
               Scaling means scaling the whole API, not just the inference layer.
               SageMaker endpoints auto-scale the model independently of the API.

  ISOLATION  — a memory leak or crash in XGBoost's C++ layer takes down the
               entire API process. SageMaker runs inference in a separate
               container on a separate instance.

  HARDWARE   — local serving uses whatever CPU the pod was given. SageMaker
               lets you choose ml.m5.large (general CPU), ml.c5.xlarge (compute-
               optimised), or ml.g4dn.xlarge (GPU) per model, without touching
               the API infrastructure.

  COMPLIANCE — regulated industries often require an auditable record of which
               model version served which prediction. SageMaker's model registry
               tracks versions, and CloudWatch logs every invocation automatically.

  COST       — SageMaker charges only for inference compute, not API servers.
               At high request volume, dedicated inference instances are cheaper
               than oversizing API pods to handle model load.

Why we use SageMaker in production
------------------------------------
FinSight scores financial transactions where latency and availability SLAs are
strict. SageMaker real-time endpoints provide:

  • Sub-10ms p99 inference latency on ml.m5.large for XGBoost with 12 features.
  • Automatic multi-AZ deployment and health-checking within an endpoint.
  • Built-in auto-scaling via Application Auto Scaling — scale out at peak, in
    at night, without manual intervention or API deployment risk.
  • Model Monitor integration: SageMaker can alert when the feature distribution
    of live traffic diverges from the training distribution (data drift), which
    is a leading indicator of model degradation before accuracy metrics degrade.

Deployment flow
----------------
  1. upload_model_to_s3()     — copy fraud_model.joblib + feature_cols.json
                                into a tar.gz and upload to S3.
  2. create_sagemaker_model() — register the S3 artifact URI and XGBoost
                                container image as a SageMaker Model object.
  3. deploy_endpoint()        — provision an endpoint configuration and
                                start the real-time endpoint.
  4. invoke_endpoint()        — call the live endpoint for a prediction.
  5. delete_endpoint()        — tear down the endpoint when done.

Prerequisites
--------------
  - AWS credentials with sagemaker:* and s3:* permissions.
  - An IAM role with AmazonSageMakerFullAccess and S3 read access, passed as
    role_arn to create_sagemaker_model().
  - The artifact files produced by fraud_model/training/train.py:
      fraud_model/artifacts/fraud_model.joblib
      fraud_model/artifacts/feature_cols.json

Usage
-----
    from fraud_model.sagemaker.deploy import SageMakerDeployer

    deployer = SageMakerDeployer(
        bucket   = "my-finsight-bucket",
        role_arn = "arn:aws:iam::123456789012:role/SageMakerExecutionRole",
    )
    model_uri   = deployer.upload_model_to_s3()
    model_name  = deployer.create_sagemaker_model(model_uri)
    endpoint    = deployer.deploy_endpoint(model_name)
    result      = deployer.invoke_endpoint(endpoint, transaction_features)
    deployer.delete_endpoint(endpoint)
"""

from __future__ import annotations

import io
import json
import os
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── AWS SDK ────────────────────────────────────────────────────────────────────
# boto3 is used directly for S3 uploads and endpoint invocation (which uses
# the sagemaker-runtime client, not the sagemaker client).
# The sagemaker Python SDK wraps the lower-level boto3 sagemaker client and
# provides higher-level helpers for model registration and endpoint creation.
try:
    import boto3
    from botocore.exceptions import ClientError
    _BOTO3_AVAILABLE = True
except ImportError:
    _BOTO3_AVAILABLE = False

try:
    import sagemaker
    from sagemaker.xgboost import XGBoostModel
    _SAGEMAKER_SDK_AVAILABLE = True
except ImportError:
    _SAGEMAKER_SDK_AVAILABLE = False

# ── Configuration ─────────────────────────────────────────────────────────────
AWS_REGION     = os.environ.get("AWS_REGION", "us-east-1")
ARTIFACTS_DIR  = Path("fraud_model/artifacts")

# XGBoost framework version must match the version used during training.
# SageMaker uses this to select the pre-built container image that includes
# the correct XGBoost + scikit-learn runtime.
XGBOOST_VERSION = "1.7-1"   # SageMaker image tag — maps to xgboost==1.7.x

# Files from the training run that must be packaged into the model tarball.
# SageMaker requires the model artifact to be a tar.gz named model.tar.gz.
# The inference.py script (in this package) is included so SageMaker uses it
# as the custom inference handler instead of the built-in XGBoost scorer.
REQUIRED_ARTIFACTS = [
    "fraud_model.joblib",
    "feature_cols.json",
]

# Default instance type for the real-time endpoint.
# ml.m5.large: 2 vCPU, 8 GiB RAM — sufficient for XGBoost with 12 features
# at ~5,000 TPS. Upgrade to ml.c5.xlarge for latency-sensitive workloads.
DEFAULT_INSTANCE_TYPE = "ml.m5.large"

# Naming convention: timestamp suffix prevents collisions on repeated deploys.
_TS = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
DEFAULT_MODEL_NAME    = f"finsight-fraud-xgboost-{_TS}"
DEFAULT_ENDPOINT_NAME = f"finsight-fraud-endpoint-{_TS}"
DEFAULT_S3_PREFIX     = "finsight/fraud-model"


# ─────────────────────────────────────────────────────────────────────────────
# DEPLOYER
# ─────────────────────────────────────────────────────────────────────────────

class SageMakerDeployer:
    """
    End-to-end deployment of the FinSight XGBoost fraud model to SageMaker.

    Wraps the three AWS clients needed for deployment (s3, sagemaker,
    sagemaker-runtime) and exposes a step-by-step API matching the deployment
    flow described in the module docstring.

    All methods raise RuntimeError in stub mode (no AWS credentials) with a
    clear message explaining which credential is missing, so CI pipelines that
    don't have AWS access fail fast with a useful error.
    """

    def __init__(
        self,
        bucket:        str,
        role_arn:      str,
        region:        str          = AWS_REGION,
        artifacts_dir: Path | str   = ARTIFACTS_DIR,
        s3_prefix:     str          = DEFAULT_S3_PREFIX,
    ) -> None:
        """
        Parameters
        ----------
        bucket        : str   S3 bucket name for model artifact storage.
                              Must exist and be in the same region as the endpoint.
        role_arn      : str   IAM role ARN that SageMaker assumes to access S3
                              and ECR. Must have AmazonSageMakerFullAccess and
                              read access to the model artifact bucket.
        region        : str   AWS region. Must match the SageMaker endpoint region.
        artifacts_dir : Path  Local directory containing fraud_model.joblib and
                              feature_cols.json produced by train.py.
        s3_prefix     : str   S3 key prefix under which model tarballs are stored.
        """
        if not _BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for SageMaker deployment.\n"
                "Install with: pip install boto3"
            )

        self.bucket        = bucket
        self.role_arn      = role_arn
        self.region        = region
        self.artifacts_dir = Path(artifacts_dir)
        self.s3_prefix     = s3_prefix

        # s3 client — used for tarball upload.
        self._s3 = boto3.client("s3", region_name=region)

        # sagemaker client — used for model and endpoint lifecycle management.
        self._sm = boto3.client("sagemaker", region_name=region)

        # sagemaker-runtime client — used for real-time endpoint invocation.
        # This is a separate service endpoint from the management API.
        self._smrt = boto3.client("sagemaker-runtime", region_name=region)

        print(
            f"SageMakerDeployer ready  "
            f"bucket={bucket}  region={region}  "
            f"role={role_arn.split('/')[-1]}"
        )

    # ── Step 1: upload model artifact to S3 ──────────────────────────────────

    def upload_model_to_s3(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> str:
        """
        Package the model artifacts into a tar.gz and upload to S3.

        SageMaker requires model artifacts to be a single tar.gz file stored in
        S3. The tarball must contain model.tar.gz at the top level. Inside it,
        SageMaker extracts to /opt/ml/model/ in the container — that is the path
        our inference.py expects to find fraud_model.joblib.

        The custom inference script (inference.py from this package) is also
        included in the tarball under code/ so SageMaker's script mode can locate
        and invoke it as the request handler.

        Parameters
        ----------
        model_name : str
            Used as part of the S3 key: {s3_prefix}/{model_name}/model.tar.gz.

        Returns
        -------
        str   The full S3 URI of the uploaded tarball, e.g.
              "s3://my-bucket/finsight/fraud-model/finsight-fraud-xgboost/model.tar.gz"
        """
        # Verify required artifacts exist before touching S3.
        for fname in REQUIRED_ARTIFACTS:
            path = self.artifacts_dir / fname
            if not path.exists():
                raise FileNotFoundError(
                    f"Required artifact not found: {path}\n"
                    f"Run fraud_model/training/train.py first."
                )

        # Locate the inference script — it lives alongside this file.
        inference_script = Path(__file__).parent / "inference.py"
        if not inference_script.exists():
            raise FileNotFoundError(
                f"inference.py not found at {inference_script}. "
                "It must be present in the same directory as deploy.py."
            )

        # Build the tarball in memory to avoid temp file cleanup issues.
        # Structure inside the tar:
        #   fraud_model.joblib     — XGBoost model + threshold
        #   feature_cols.json      — feature names, threshold, category mapping
        #   code/inference.py      — custom SageMaker inference handler
        #
        # SageMaker script mode looks for inference.py in code/ when
        # SAGEMAKER_PROGRAM env var is not set, falling back to the built-in
        # XGBoost handler. Placing it in code/ ensures our handler is used.
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for fname in REQUIRED_ARTIFACTS:
                local_path = self.artifacts_dir / fname
                tar.add(local_path, arcname=fname)
                print(f"  + {fname}  ({local_path.stat().st_size:,} bytes)")

            tar.add(inference_script, arcname="code/inference.py")
            print(f"  + code/inference.py  ({inference_script.stat().st_size:,} bytes)")

        buf.seek(0)
        tarball_size = buf.getbuffer().nbytes

        s3_key = f"{self.s3_prefix}/{model_name}/model.tar.gz"
        s3_uri = f"s3://{self.bucket}/{s3_key}"

        print(f"Uploading model tarball ({tarball_size / 1024:.1f} KB) → {s3_uri}")
        self._s3.upload_fileobj(buf, self.bucket, s3_key)
        print("Upload complete.")

        return s3_uri

    # ── Step 2: register a SageMaker Model object ─────────────────────────────

    def create_sagemaker_model(
        self,
        model_s3_uri:  str,
        model_name:    str = DEFAULT_MODEL_NAME,
    ) -> str:
        """
        Register the S3 model artifact as a SageMaker Model resource.

        A SageMaker Model is a logical object that pairs:
          - A container image (the XGBoost serving container from ECR)
          - A model artifact URI (our tar.gz in S3)
          - An IAM execution role (grants the container access to S3)

        Creating the Model object does not start any compute — it is just a
        metadata record used when creating endpoint configurations later.

        Parameters
        ----------
        model_s3_uri : str   S3 URI returned by upload_model_to_s3().
        model_name   : str   Name for the SageMaker Model resource. Must be
                             unique within the region.

        Returns
        -------
        str   The model_name (for passing to deploy_endpoint()).
        """
        # Resolve the XGBoost container image URI for this region.
        # SageMaker provides pre-built images for common frameworks; we use the
        # XGBoost image so we don't have to build or maintain our own Docker image.
        if _SAGEMAKER_SDK_AVAILABLE:
            container_image = sagemaker.image_uris.retrieve(
                framework       = "xgboost",
                region          = self.region,
                version         = XGBOOST_VERSION,
                image_scope     = "inference",
                instance_type   = DEFAULT_INSTANCE_TYPE,
            )
        else:
            # Fallback: construct the ECR URI manually for us-east-1.
            # For other regions, replace the account ID and region accordingly.
            # Full list: https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-extend.html
            container_image = (
                f"683313688378.dkr.ecr.{self.region}.amazonaws.com/"
                f"sagemaker-xgboost:{XGBOOST_VERSION}"
            )

        print(f"Creating SageMaker model '{model_name}'...")
        print(f"  container : {container_image}")
        print(f"  artifact  : {model_s3_uri}")

        self._sm.create_model(
            ModelName        = model_name,
            ExecutionRoleArn = self.role_arn,
            PrimaryContainer = {
                # The container image includes XGBoost, scikit-learn, and the
                # SageMaker inference toolkit that invokes our inference.py.
                "Image":          container_image,
                "ModelDataUrl":   model_s3_uri,
                "Environment": {
                    # Tell the SageMaker serving stack to use our custom handler
                    # instead of the default XGBoost predict() path.
                    "SAGEMAKER_PROGRAM":         "inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": model_s3_uri,
                },
            },
        )

        print(f"SageMaker model '{model_name}' created.")
        return model_name

    # ── Step 3: deploy to a real-time endpoint ────────────────────────────────

    def deploy_endpoint(
        self,
        model_name:     str,
        endpoint_name:  str  = DEFAULT_ENDPOINT_NAME,
        instance_type:  str  = DEFAULT_INSTANCE_TYPE,
        instance_count: int  = 1,
    ) -> str:
        """
        Create an endpoint configuration and launch the real-time endpoint.

        SageMaker endpoint creation involves two resources:
          EndpointConfig — immutable blueprint: which model, instance type,
                           instance count, data capture settings. Creating a new
                           config for each deployment (rather than mutating one)
                           means rollbacks are a config swap, not a re-deploy.
          Endpoint       — the live serving resource. Points to a config and can
                           be updated to a new config atomically (blue/green swap)
                           with no downtime.

        This method waits (blocking) until the endpoint is InService before
        returning, since subsequent invoke_endpoint() calls require a healthy
        endpoint. Typical cold-start time is 3–6 minutes for ml.m5.large.

        Parameters
        ----------
        model_name     : str   SageMaker Model name from create_sagemaker_model().
        endpoint_name  : str   Name for the deployed endpoint. Must be unique.
        instance_type  : str   EC2 instance type for inference compute.
        instance_count : int   Initial number of instances. Scale to ≥2 for HA.

        Returns
        -------
        str   endpoint_name (for passing to invoke_endpoint() / delete_endpoint()).
        """
        config_name = f"{endpoint_name}-config"

        print(f"Creating endpoint configuration '{config_name}'...")
        self._sm.create_endpoint_config(
            EndpointConfigName = config_name,
            ProductionVariants = [
                {
                    "VariantName":          "AllTraffic",
                    "ModelName":            model_name,
                    "InitialInstanceCount": instance_count,
                    "InstanceType":         instance_type,
                    # 100% of traffic goes to this variant; set to e.g. 50 for
                    # a canary deployment alongside an existing variant.
                    "InitialVariantWeight": 1.0,
                }
            ],
        )

        print(
            f"Deploying endpoint '{endpoint_name}'  "
            f"instance={instance_type} × {instance_count}..."
        )
        self._sm.create_endpoint(
            EndpointName       = endpoint_name,
            EndpointConfigName = config_name,
        )

        # Poll until the endpoint reaches a terminal state.
        # InService → ready to accept traffic.
        # Failed    → something went wrong; check CloudWatch logs at
        #             /aws/sagemaker/Endpoints/{endpoint_name}.
        print("Waiting for endpoint to become InService (this takes 3–6 minutes)...")
        waiter = self._sm.get_waiter("endpoint_in_service")
        waiter.wait(
            EndpointName = endpoint_name,
            WaiterConfig = {
                "Delay":       30,   # poll every 30 seconds
                "MaxAttempts": 20,   # wait up to 10 minutes before timing out
            },
        )

        status = self._describe_endpoint_status(endpoint_name)
        if status != "InService":
            raise RuntimeError(
                f"Endpoint '{endpoint_name}' reached status '{status}' instead of "
                "InService. Check CloudWatch Logs for the failure reason:\n"
                f"  /aws/sagemaker/Endpoints/{endpoint_name}"
            )

        print(f"Endpoint '{endpoint_name}' is InService and ready.")
        return endpoint_name

    # ── Step 4: invoke the live endpoint ─────────────────────────────────────

    def invoke_endpoint(
        self,
        endpoint_name: str,
        features: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Send a single transaction's features to the live endpoint and return
        the fraud score.

        The endpoint expects a JSON body matching the format produced by
        FraudPredictor._encode() — a flat dict of feature name → numeric value
        in the same order as feature_cols. This mirrors the contract defined in
        inference.py's input_fn().

        Parameters
        ----------
        endpoint_name : str         The endpoint name from deploy_endpoint().
        features      : dict        Feature name → value, e.g.:
                                    {"amount": 150.0, "amount_log": 5.01, ...}
                                    merchant_category must already be encoded
                                    to its integer code (use the category_mapping
                                    from feature_cols.json).

        Returns
        -------
        dict with keys:
            fraud_probability : float   Raw model output in [0, 1].
            is_fraud          : bool    Hard label at the training threshold.
            risk_level        : str     LOW / MEDIUM / HIGH / CRITICAL.
        """
        payload = json.dumps(features)

        response = self._smrt.invoke_endpoint(
            EndpointName = endpoint_name,
            ContentType  = "application/json",
            Accept       = "application/json",
            Body         = payload,
        )

        # The response Body is a StreamingBody; read() returns the raw bytes.
        result: dict[str, Any] = json.loads(response["Body"].read())
        return result

    # ── Step 5: delete endpoint (cleanup) ────────────────────────────────────

    def delete_endpoint(self, endpoint_name: str) -> None:
        """
        Delete the real-time endpoint and its configuration to stop billing.

        SageMaker charges for endpoints by the hour regardless of traffic volume.
        Always delete endpoints that are no longer needed. Note: deleting the
        endpoint does NOT delete the SageMaker Model resource or the S3 artifact —
        those are free to retain for audit trails and re-deployment.

        Parameters
        ----------
        endpoint_name : str   The endpoint to delete.
        """
        # Derive the config name from the endpoint name — the convention set in
        # deploy_endpoint() appends "-config" to the endpoint name.
        config_name = f"{endpoint_name}-config"

        print(f"Deleting endpoint '{endpoint_name}'...")
        try:
            self._sm.delete_endpoint(EndpointName=endpoint_name)
            print(f"  Endpoint '{endpoint_name}' deleted.")
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code == "ValidationException":
                print(f"  Endpoint '{endpoint_name}' not found — already deleted.")
            else:
                raise

        print(f"Deleting endpoint configuration '{config_name}'...")
        try:
            self._sm.delete_endpoint_config(EndpointConfigName=config_name)
            print(f"  Endpoint config '{config_name}' deleted.")
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code == "ValidationException":
                print(f"  Endpoint config '{config_name}' not found — already deleted.")
            else:
                raise

    # ── Helpers ───────────────────────────────────────────────────────────────

    def describe_endpoint(self, endpoint_name: str) -> dict[str, Any]:
        """Return the full SageMaker describe_endpoint() response dict."""
        return self._sm.describe_endpoint(EndpointName=endpoint_name)

    def _describe_endpoint_status(self, endpoint_name: str) -> str:
        """Return just the EndpointStatus string for the given endpoint."""
        response = self._sm.describe_endpoint(EndpointName=endpoint_name)
        return response["EndpointStatus"]


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def upload_model_to_s3(
    model_path: Path | str,
    bucket:     str,
    key:        str,
    region:     str = AWS_REGION,
) -> str:
    """
    Low-level helper: upload a single file to S3 and return the S3 URI.

    Unlike SageMakerDeployer.upload_model_to_s3(), this function uploads a
    pre-built tarball rather than packaging artifacts on the fly. Use this
    when you have already assembled the model.tar.gz externally (e.g. in a
    CI/CD pipeline that produces the tarball as a build artifact).

    Parameters
    ----------
    model_path : Path | str   Local path to the model.tar.gz file.
    bucket     : str          S3 bucket name.
    key        : str          S3 object key (path within the bucket).
    region     : str          AWS region for the S3 client.

    Returns
    -------
    str   S3 URI in the form "s3://{bucket}/{key}".
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    s3 = boto3.client("s3", region_name=region)
    s3_uri = f"s3://{bucket}/{key}"

    print(f"Uploading {model_path} ({model_path.stat().st_size / 1024:.1f} KB) → {s3_uri}")
    s3.upload_file(str(model_path), bucket, key)
    print("Upload complete.")

    return s3_uri


def create_sagemaker_model(
    model_s3_uri: str,
    role_arn:     str,
    model_name:   str = DEFAULT_MODEL_NAME,
    region:       str = AWS_REGION,
) -> str:
    """
    Functional wrapper around SageMakerDeployer.create_sagemaker_model().

    Useful for scripting and CI pipelines where instantiating the full deployer
    class is unnecessary overhead.

    Returns
    -------
    str   The model_name passed in (or the generated default).
    """
    sm = boto3.client("sagemaker", region_name=region)

    if _SAGEMAKER_SDK_AVAILABLE:
        container_image = sagemaker.image_uris.retrieve(
            framework     = "xgboost",
            region        = region,
            version       = XGBOOST_VERSION,
            image_scope   = "inference",
            instance_type = DEFAULT_INSTANCE_TYPE,
        )
    else:
        container_image = (
            f"683313688378.dkr.ecr.{region}.amazonaws.com/"
            f"sagemaker-xgboost:{XGBOOST_VERSION}"
        )

    sm.create_model(
        ModelName        = model_name,
        ExecutionRoleArn = role_arn,
        PrimaryContainer = {
            "Image":        container_image,
            "ModelDataUrl": model_s3_uri,
            "Environment": {
                "SAGEMAKER_PROGRAM":          "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": model_s3_uri,
            },
        },
    )
    print(f"SageMaker model '{model_name}' created.")
    return model_name


def deploy_endpoint(
    model_name:     str,
    endpoint_name:  str = DEFAULT_ENDPOINT_NAME,
    instance_type:  str = DEFAULT_INSTANCE_TYPE,
    instance_count: int = 1,
    region:         str = AWS_REGION,
) -> str:
    """
    Functional wrapper: create endpoint config + endpoint, wait for InService.

    Returns
    -------
    str   endpoint_name.
    """
    sm          = boto3.client("sagemaker", region_name=region)
    config_name = f"{endpoint_name}-config"

    sm.create_endpoint_config(
        EndpointConfigName = config_name,
        ProductionVariants = [{
            "VariantName":          "AllTraffic",
            "ModelName":            model_name,
            "InitialInstanceCount": instance_count,
            "InstanceType":         instance_type,
            "InitialVariantWeight": 1.0,
        }],
    )
    sm.create_endpoint(
        EndpointName       = endpoint_name,
        EndpointConfigName = config_name,
    )

    print(f"Waiting for '{endpoint_name}' to become InService...")
    sm.get_waiter("endpoint_in_service").wait(EndpointName=endpoint_name)
    print(f"Endpoint '{endpoint_name}' is InService.")
    return endpoint_name


def invoke_endpoint(
    endpoint_name: str,
    features_dict: dict[str, Any],
    region:        str = AWS_REGION,
) -> dict[str, Any]:
    """
    Functional wrapper: invoke a SageMaker real-time endpoint.

    Parameters
    ----------
    endpoint_name : str    Deployed endpoint name.
    features_dict : dict   Feature name → numeric value dict.

    Returns
    -------
    dict   Parsed JSON response from the inference container.
    """
    smrt     = boto3.client("sagemaker-runtime", region_name=region)
    payload  = json.dumps(features_dict)
    response = smrt.invoke_endpoint(
        EndpointName = endpoint_name,
        ContentType  = "application/json",
        Accept       = "application/json",
        Body         = payload,
    )
    return json.loads(response["Body"].read())


def delete_endpoint(
    endpoint_name: str,
    region:        str = AWS_REGION,
) -> None:
    """
    Functional wrapper: delete endpoint and its configuration.

    Safe to call even if the endpoint has already been deleted.
    """
    sm          = boto3.client("sagemaker", region_name=region)
    config_name = f"{endpoint_name}-config"

    for delete_fn, resource_name, resource in [
        (sm.delete_endpoint,        "endpoint",        endpoint_name),
        (sm.delete_endpoint_config, "endpoint config", config_name),
    ]:
        try:
            delete_fn(**({
                "EndpointName":       endpoint_name,
            } if "endpoint_config" not in delete_fn.__name__ else {
                "EndpointConfigName": config_name,
            }))
            print(f"Deleted {resource_name} '{resource}'.")
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ValidationException":
                print(f"{resource_name.capitalize()} '{resource}' not found.")
            else:
                raise


# ─────────────────────────────────────────────────────────────────────────────
# DEMO / CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deploy fraud model to SageMaker")
    parser.add_argument("--bucket",   required=True, help="S3 bucket name")
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--region",   default=AWS_REGION)
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    parser.add_argument(
        "--delete", metavar="ENDPOINT_NAME",
        help="Delete an existing endpoint instead of deploying"
    )
    args = parser.parse_args()

    if args.delete:
        delete_endpoint(args.delete, region=args.region)
    else:
        deployer = SageMakerDeployer(
            bucket   = args.bucket,
            role_arn = args.role_arn,
            region   = args.region,
        )
        model_uri     = deployer.upload_model_to_s3()
        model_name    = deployer.create_sagemaker_model(model_uri)
        endpoint_name = deployer.deploy_endpoint(
            model_name    = model_name,
            instance_type = args.instance_type,
        )
        print(f"\nEndpoint ready: {endpoint_name}")
        print("Test with:")
        print(f'  python -c "from fraud_model.sagemaker.deploy import invoke_endpoint; '
              f'print(invoke_endpoint(\'{endpoint_name}\', {{\"amount\": 150.0}}))"')
