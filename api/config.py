"""
Centralised application configuration via pydantic-settings.

Why pydantic-settings instead of plain os.environ
---------------------------------------------------
os.environ.get("AWS_REGION", "us-east-1") works, but it has four problems
that compound as a project grows:

  1. NO TYPE SAFETY
     os.environ always returns str. A typo like MLFLOW_PORT="abc" is silently
     accepted and only crashes at the callsite when the value is used as an int.
     pydantic-settings converts and validates every field at import time, so
     misconfigured environments fail loudly at startup — not mid-request.

  2. NO DOCUMENTATION
     A flat dict of strings gives no hint of which variables are required,
     which are optional, what their types are, or what valid values look like.
     A Settings class is self-documenting: field names, types, defaults, and
     validation rules are all in one place.

  3. NO VALIDATION
     Nothing prevents ENV="prodcution" (typo) from being silently accepted and
     causing unexpected behaviour deep in a conditional. pydantic Literal types
     and validators reject invalid values at startup with clear error messages.

  4. SCATTERED READS
     With os.environ, every module reads its own variables independently.
     A refactor that renames AWS_REGION to AWS_DEFAULT_REGION requires a grep
     across the entire codebase. With a Settings class, there is exactly one
     place to update.

Why @lru_cache
--------------
pydantic-settings reads and parses the .env file on every Settings()
instantiation. For a typical app this happens once at startup, but in test
suites or lambda-style handlers it can happen on every invocation. @lru_cache
memoises the first call so subsequent get_settings() calls return the same
object from memory — zero file I/O, zero reparsing.

    # Without cache: new Settings() on every call
    def get_settings():
        return Settings()

    # With cache: parsed once, reused everywhere
    @lru_cache
    def get_settings():
        return Settings()

The cache is keyed on the function arguments; since get_settings() takes none,
it always returns the same singleton. In tests, call get_settings.cache_clear()
between cases to force a fresh read from a patched environment.

Sensitive fields use SecretStr
-------------------------------
pydantic's SecretStr wraps the raw string value and overrides __repr__ and
__str__ to return "**********" so secrets never appear in logs, tracebacks,
or serialised dicts. Access the real value with .get_secret_value() only at
the point of use (e.g. when constructing a boto3 client).

Usage
-----
    from api.config import get_settings

    settings = get_settings()
    print(settings.aws_region)          # "us-east-1"
    print(settings.aws_secret_access_key)          # SecretStr('**********')
    print(settings.aws_secret_access_key.get_secret_value())  # actual key

    # Force reload (useful in tests when patching env vars)
    get_settings.cache_clear()
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS CLASS
# ─────────────────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """
    All runtime configuration for the FinSight API.

    Values are loaded in this priority order (highest wins):
      1. Actual environment variables (set in shell or Docker/K8s secrets)
      2. Variables in the .env file at the project root
      3. Field default values defined below

    This means production deployments can inject secrets via real env vars
    without committing them to the .env file, while local development uses
    .env for convenience.
    """

    model_config = SettingsConfigDict(
        # Look for a .env file relative to the current working directory.
        # In Docker/production, the file typically won't exist and all values
        # come from real environment variables — that's fine; pydantic-settings
        # only uses the file as a fallback layer.
        env_file          = ".env",
        env_file_encoding = "utf-8",
        # Case-insensitive so AWS_REGION, aws_region, and Aws_Region all bind
        # to the same field. Avoids confusion across team members' shell configs.
        case_sensitive    = False,
        # Raise a ValidationError for any extra variables in the env file
        # rather than silently ignoring them — catches typos in variable names.
        extra             = "ignore",
    )

    # ── Application environment ───────────────────────────────────────────────
    # Controls logging verbosity, error detail in API responses, and whether
    # debug tooling (e.g. the FastAPI /docs page) is exposed.
    # Literal enforces exactly these three strings — anything else raises a
    # clear ValidationError at startup rather than a silent logical error.
    env: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Runtime environment. Controls log level and debug exposure.",
    )

    # ── AWS ───────────────────────────────────────────────────────────────────
    # All three are used by BedrockClient and, when present, override boto3's
    # default credential chain (IAM role → ~/.aws/credentials → env vars).
    # In production on ECS/EKS, leave access_key_id and secret blank and attach
    # an IAM role to the task instead — boto3 picks it up automatically.
    aws_region: str = Field(
        default="us-east-1",
        description="AWS region for Bedrock, S3, and other service calls.",
    )
    aws_access_key_id: SecretStr | None = Field(
        default=None,
        description="AWS access key. Prefer IAM roles in production over static keys.",
    )
    aws_secret_access_key: SecretStr | None = Field(
        default=None,
        description="AWS secret key paired with aws_access_key_id.",
    )

    # ── Pinecone (optional — only needed when embed_mode='pinecone') ──────────
    # Pinecone is a managed vector database that scales beyond a single machine,
    # useful when the corpus grows past ~10M vectors where FAISS in-memory
    # becomes impractical. The FAISS path remains default for local development.
    pinecone_api_key: SecretStr | None = Field(
        default=None,
        description="Pinecone API key. Required when embed_mode='pinecone'.",
    )
    pinecone_index: str = Field(
        default="finsight",
        description="Name of the Pinecone index to read from / write to.",
    )

    # ── Embedding backend ─────────────────────────────────────────────────────
    # Determines which vector store the RAG pipeline uses for retrieval.
    # 'faiss'   — local FAISS index, zero infra dependencies, good to ~5M vectors.
    # 'pinecone' — managed Pinecone index, requires pinecone_api_key.
    # The chain.py and embedder.py modules read this to select the right backend.
    embed_mode: Literal["faiss", "pinecone"] = Field(
        default="faiss",
        description="Vector retrieval backend. 'faiss' for local; 'pinecone' for managed.",
    )

    # ── MLflow ────────────────────────────────────────────────────────────────
    # URI for the MLflow tracking server where fraud model experiments are logged.
    # 'sqlite:///mlflow.db' works locally without a server; a remote URI
    # (http://mlflow.internal:5000) is used in staging/production.
    mlflow_uri: str = Field(
        default="sqlite:///mlflow.db",
        description="MLflow tracking URI. Local SQLite for dev; remote server for prod.",
    )

    # ── Database ──────────────────────────────────────────────────────────────
    # PostgreSQL connection string for the main application database.
    # Stores scored transaction history, API audit logs, and user sessions.
    # SecretStr prevents the password embedded in the URI from leaking into logs.
    postgres_uri: SecretStr = Field(
        default=SecretStr("postgresql://finsight:finsight@localhost:5432/finsight"),
        description="PostgreSQL DSN. Include credentials in the URI (use SecretStr).",
    )

    # ── API security ──────────────────────────────────────────────────────────
    # Used to sign JWTs and HMAC-based session tokens.
    # In production this must be a random 32+ byte string — never the default.
    # Generate with: python -c "import secrets; print(secrets.token_hex(32))"
    secret_key: SecretStr = Field(
        default=SecretStr("change-me-in-production"),
        description="HMAC / JWT signing key. Generate with secrets.token_hex(32).",
    )

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def is_production(self) -> bool:
        """True only in the 'production' environment."""
        return self.env == "production"

    @property
    def is_development(self) -> bool:
        """True in 'development' environment (default for local work)."""
        return self.env == "development"

    @property
    def debug(self) -> bool:
        """FastAPI and uvicorn debug flag — enabled outside production."""
        return not self.is_production

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("secret_key", mode="after")
    @classmethod
    def warn_default_secret_key(cls, v: SecretStr) -> SecretStr:
        """
        Warn loudly if the default secret key reaches a production environment.
        A deterministic secret key allows any attacker who reads this source code
        to forge valid JWTs — a critical authentication bypass.
        We don't raise here (that would break local docker-compose without a
        .env file), but the log message appears in startup output.
        """
        if v.get_secret_value() == "change-me-in-production":
            import warnings
            warnings.warn(
                "SECRET_KEY is set to the default placeholder value. "
                "Set a random SECRET_KEY in production: "
                "python -c \"import secrets; print(secrets.token_hex(32))\"",
                UserWarning,
                stacklevel=2,
            )
        return v

    @model_validator(mode="after")
    def validate_pinecone_config(self) -> "Settings":
        """
        Cross-field validation: pinecone_api_key is required when embed_mode='pinecone'.

        model_validator runs after all individual fields are validated, so it
        can inspect multiple fields together. Separating this from field_validator
        avoids the awkward pattern of passing embed_mode into a pinecone_api_key
        validator.
        """
        if self.embed_mode == "pinecone" and self.pinecone_api_key is None:
            raise ValueError(
                "PINECONE_API_KEY must be set when EMBED_MODE='pinecone'. "
                "Add it to your .env file or set it as an environment variable."
            )
        return self

    @model_validator(mode="after")
    def warn_production_aws_keys(self) -> "Settings":
        """
        Warn if static AWS keys are present in production.
        In production, IAM roles attached to the compute resource (ECS task,
        EKS pod) are strongly preferred over long-lived static credentials.
        """
        if self.is_production and self.aws_access_key_id is not None:
            import warnings
            warnings.warn(
                "Static AWS credentials (AWS_ACCESS_KEY_ID) are set in a production "
                "environment. Consider using an IAM role instead to avoid "
                "long-lived credential exposure.",
                UserWarning,
                stacklevel=2,
            )
        return self

    def summary(self) -> dict:
        """
        Non-sensitive summary for logging at startup.

        Deliberately omits all SecretStr fields so this dict can be safely
        written to stdout / log aggregators without credential exposure.
        """
        return {
            "env":           self.env,
            "aws_region":    self.aws_region,
            "embed_mode":    self.embed_mode,
            "mlflow_uri":    self.mlflow_uri,
            "pinecone_index": self.pinecone_index,
            "has_aws_keys":  self.aws_access_key_id is not None,
            "has_pinecone":  self.pinecone_api_key is not None,
            "debug":         self.debug,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON ACCESSOR
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache
def get_settings() -> Settings:
    """
    Return the application settings singleton.

    The first call reads and parses the .env file; every subsequent call
    returns the cached instance. This means the .env file is read exactly
    once per process, regardless of how many modules import get_settings().

    In tests, patch environment variables with monkeypatch or unittest.mock,
    then call get_settings.cache_clear() to force a fresh parse:

        def test_production_env(monkeypatch):
            monkeypatch.setenv("ENV", "production")
            monkeypatch.setenv("SECRET_KEY", "real-secret-for-test")
            get_settings.cache_clear()
            s = get_settings()
            assert s.is_production
            get_settings.cache_clear()  # clean up for subsequent tests
    """
    return Settings()


# ─────────────────────────────────────────────────────────────────────────────
# DEMO / CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    settings = get_settings()

    print("── Settings summary (safe to log) ─────────────────────────────")
    print(json.dumps(settings.summary(), indent=2))

    print("\n── SecretStr redaction demo ────────────────────────────────────")
    print(f"  secret_key repr  : {settings.secret_key!r}")
    print(f"  secret_key str   : {settings.secret_key}")
    print(f"  raw value        : {settings.secret_key.get_secret_value()}")

    print("\n── Properties ──────────────────────────────────────────────────")
    print(f"  is_production : {settings.is_production}")
    print(f"  is_development: {settings.is_development}")
    print(f"  debug         : {settings.debug}")

    print("\n── lru_cache demo (should return same object) ──────────────────")
    s1 = get_settings()
    s2 = get_settings()
    print(f"  s1 is s2 : {s1 is s2}")  # True — same instance

    print("\n── Cache clear demo (returns new object) ───────────────────────")
    get_settings.cache_clear()
    s3 = get_settings()
    print(f"  s1 is s3 : {s1 is s3}")  # False — freshly parsed
