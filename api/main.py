"""
FinSight AI Platform — FastAPI application entry point.

This module is the outermost layer of the API. Its responsibilities are:
  1. Create and configure the FastAPI application.
  2. Register middleware that wraps every incoming request.
  3. Mount routers that define the actual endpoints.
  4. Manage startup and shutdown of shared, expensive resources.

Everything below is intentionally thin: no business logic lives here.
Each router owns the logic for its domain; this file just wires them together.

Running the server
------------------
    # Development (auto-reload on file changes)
    uvicorn api.main:app --reload --port 8000

    # Production (multiple workers, no reload)
    uvicorn api.main:app --workers 4 --port 8000

    # Via Docker (see infrastructure/docker/Dockerfile.api)
    docker compose up api
"""

from __future__ import annotations

import logging
import time
import warnings
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import get_settings

logger = logging.getLogger(__name__)


# ── Router imports (graceful degradation during incremental development) ───────
# Each router module is imported inside a try/except so that the server can
# start and serve the health and root endpoints even while other routers are
# still being built. A warning is printed for any missing router so developers
# know which routes are not yet active without reading a stack trace.
#
# In production, all four routers must be present — the health router is the
# liveness/readiness probe used by Kubernetes and Docker healthchecks.

_missing_routers: list[str] = []

try:
    from api.routers import health as health_router
except ImportError as _e:
    warnings.warn(f"health router unavailable: {_e}", ImportWarning, stacklevel=1)
    health_router = None  # type: ignore[assignment]
    _missing_routers.append("health")

try:
    from api.routers import fraud as fraud_router
except ImportError as _e:
    warnings.warn(f"fraud router unavailable: {_e}", ImportWarning, stacklevel=1)
    fraud_router = None  # type: ignore[assignment]
    _missing_routers.append("fraud")

try:
    from api.routers import query as query_router
except ImportError as _e:
    warnings.warn(f"query router unavailable: {_e}", ImportWarning, stacklevel=1)
    query_router = None  # type: ignore[assignment]
    _missing_routers.append("query")

try:
    from api.routers import explain as explain_router
except ImportError as _e:
    warnings.warn(f"explain router unavailable: {_e}", ImportWarning, stacklevel=1)
    explain_router = None  # type: ignore[assignment]
    _missing_routers.append("explain")


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage startup and shutdown of shared, long-lived resources.

    FastAPI's lifespan replaces the old @app.on_event("startup") decorators.
    Everything before `yield` runs once when the server starts; everything
    after `yield` runs once when the server receives a shutdown signal (SIGTERM).

    Why load models here instead of at module import time?
    - Module-level loading runs when the module is first imported — including
      during test collection, CLI --help, linting, and type-checking runs where
      the model weights should NOT be loaded.
    - Lifespan loading is explicit and controllable: it only happens when a
      server process is actually starting up.
    - Resources stored on app.state are accessible to every request handler via
      `request.app.state.embedder` without global variables.

    What lives on app.state:
      app.state.chain      — FinSightChain (FAISS + Bedrock, shared across requests)
      app.state.predictor  — FraudPredictor (XGBoost model, shared across requests)
      app.state.started_at — ISO timestamp of when the server started
    """
    settings = get_settings()

    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info("Starting FinSight AI Platform (env=%s)", settings.env)
    logger.info("Configuration: %s", settings.summary())

    app.state.started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Load the RAG chain. FAISSEmbedder reads model weights (~400 MB) and the
    # FAISS index from disk. BedrockClient resolves AWS credentials. Both are
    # expensive to initialise but cheap to reuse — load once, reuse per request.
    try:
        from rag_pipeline.chain import FinSightChain
        app.state.chain = FinSightChain()
        logger.info(
            "FinSightChain loaded  vectors=%d  stub=%s",
            app.state.chain.embedder.vector_count,
            app.state.chain.bedrock._stub_mode,
        )
    except Exception as exc:
        # Chain load failure is non-fatal at startup — the /fraud and /health
        # endpoints still work. Log the error and store None so routers that
        # depend on the chain can return a 503 rather than crashing.
        logger.warning("FinSightChain failed to load: %s", exc)
        app.state.chain = None

    # Load the fraud model. joblib.load reads ~50 MB of XGBoost weights and
    # the SHAP explainer from disk. Like the chain, loaded once at startup.
    try:
        from fraud_model.serving.predictor import FraudPredictor
        app.state.predictor = FraudPredictor()
        logger.info(
            "FraudPredictor loaded  features=%d  threshold=%.3f",
            len(app.state.predictor.feature_cols),
            app.state.predictor.threshold,
        )
    except Exception as exc:
        logger.warning("FraudPredictor failed to load: %s", exc)
        app.state.predictor = None

    if _missing_routers:
        logger.warning("Routers not yet available: %s", _missing_routers)

    logger.info("FinSight API ready — http://0.0.0.0:8000")

    yield   # ← server is live and handling requests between here and shutdown

    # ── Shutdown ──────────────────────────────────────────────────────────────
    # Clean up resources that hold OS-level handles or external connections.
    # Python's garbage collector handles memory, but explicit cleanup avoids
    # "connection was closed in the middle of a query" errors in PostgreSQL
    # connection pools and unclosed socket warnings in test output.
    logger.info("Shutting down FinSight AI Platform...")
    app.state.chain      = None
    app.state.predictor  = None
    logger.info("Shutdown complete.")


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "FinSight AI Platform",
    version     = "1.0.0",
    description = (
        "ML-powered financial intelligence platform providing real-time fraud "
        "detection, SHAP-based transaction explanations, and RAG-assisted "
        "compliance Q&A over SEC filings."
    ),
    # Expose OpenAPI docs in development; disable in production to reduce the
    # attack surface (the /docs and /redoc endpoints don't require auth).
    docs_url    = "/docs"    if get_settings().debug else None,
    redoc_url   = "/redoc"   if get_settings().debug else None,
    openapi_url = "/openapi.json" if get_settings().debug else None,
    lifespan    = lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# MIDDLEWARE
# ─────────────────────────────────────────────────────────────────────────────

# ── CORS ──────────────────────────────────────────────────────────────────────
# CORS (Cross-Origin Resource Sharing) is a browser security mechanism that
# blocks JavaScript in one origin (e.g. https://dashboard.finsight.io) from
# reading responses from a different origin (https://api.finsight.io) unless
# the server explicitly opts in via these headers.
#
# allow_origins controls which frontend URLs may call this API:
#   - In development, "*" allows any origin so engineers can run the frontend
#     on any port (localhost:3000, :5173, etc.) without reconfiguring the API.
#   - In production, restrict to the exact deployed frontend domain(s) to
#     prevent cross-site request forgery from malicious third-party pages.
#
# allow_credentials=True permits cookies and Authorization headers to be sent
# with cross-origin requests — required for JWT-based auth flows.
#
# allow_methods and allow_headers control the HTTP verbs and headers the
# browser may include in preflight (OPTIONS) requests. "*" means unrestricted.
_settings = get_settings()
_cors_origins = (
    ["*"]
    if _settings.is_development
    else [
        "https://finsight.io",
        "https://dashboard.finsight.io",
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = _cors_origins,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Request timing middleware ─────────────────────────────────────────────────
# Adds an X-Process-Time header to every response showing how long the request
# took in milliseconds. Useful for latency profiling in dashboards and for
# spotting slow endpoints during load testing without a full APM tool.
# Implemented as a raw ASGI middleware (via @app.middleware) rather than a
# Starlette BaseHTTPMiddleware subclass to avoid the double-body-read issue
# that affects streaming responses in some BaseHTTPMiddleware implementations.
@app.middleware("http")
async def add_process_time_header(request: Request, call_next) -> Response:
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time"] = f"{elapsed_ms:.2f}ms"
    return response


# ─────────────────────────────────────────────────────────────────────────────
# ROUTERS
# ─────────────────────────────────────────────────────────────────────────────
# Each router is a self-contained FastAPI APIRouter that owns a domain area.
# Mounting them here with a prefix keeps URL paths predictable and ensures
# OpenAPI groups endpoints by domain in the /docs UI.
#
# /health  — liveness and readiness probes for Kubernetes / Docker healthchecks.
#            No auth required so the orchestrator can reach it unconditionally.
#
# /fraud   — real-time transaction scoring (score, batch_score) and risk
#            summaries. Reads from app.state.predictor (FraudPredictor).
#
# /query   — RAG compliance Q&A over ingested SEC filings. Accepts a natural-
#            language question, runs the full FinSightChain pipeline, and returns
#            a grounded answer with citation labels and quality scores.
#            Reads from app.state.chain (FinSightChain).
#
# /explain — SHAP-based explanations for previously scored transactions.
#            Given a transaction_id, retrieves the stored ScoreResult from
#            FraudPredictor and generates a plain-English breakdown via Claude.
#            Reads from both app.state.predictor and app.state.chain.

if health_router is not None:
    app.include_router(health_router.router, prefix="/health", tags=["Health"])

if fraud_router is not None:
    app.include_router(fraud_router.router, prefix="/fraud",  tags=["Fraud Detection"])

if query_router is not None:
    app.include_router(query_router.router, prefix="/query",  tags=["Compliance Q&A"])

if explain_router is not None:
    app.include_router(explain_router.router, prefix="/explain", tags=["Explainability"])


# ─────────────────────────────────────────────────────────────────────────────
# ROOT ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/",
    summary     = "Service info",
    description = "Returns platform metadata and a map of available API endpoints.",
    response_model = None,
    tags        = ["Meta"],
)
async def root(request: Request) -> JSONResponse:
    """
    Discovery endpoint — returns service identity, version, and endpoint map.

    Intended for:
      - Health dashboards that need to confirm the right service is running.
      - New engineers or API consumers who want a quick overview without
        opening the /docs UI.
      - Load balancer health checks that expect a 200 on /.

    The endpoint map is generated from the app's registered routes at request
    time so it always reflects the currently mounted routers, including any
    that were conditionally excluded due to missing router modules.
    """
    settings = get_settings()

    # Build the endpoint catalogue from the live route table rather than
    # hardcoding paths, so newly added routes appear automatically.
    endpoints: dict[str, Any] = {}
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            path = route.path
            if path in ("/", "/openapi.json"):
                continue
            for method in sorted(route.methods or []):
                key = f"{method} {path}"
                endpoints[key] = getattr(route, "summary", path)

    chain_status     = None
    predictor_status = None
    if hasattr(request.app.state, "chain") and request.app.state.chain is not None:
        chain_status = request.app.state.chain.status()
    if hasattr(request.app.state, "predictor") and request.app.state.predictor is not None:
        predictor_status = {
            "features":  len(request.app.state.predictor.feature_cols),
            "threshold": round(request.app.state.predictor.threshold, 4),
            "store_size": request.app.state.predictor.store_size(),
        }

    return JSONResponse({
        "service":     "FinSight AI Platform",
        "version":     app.version,
        "environment": settings.env,
        "started_at":  getattr(request.app.state, "started_at", None),
        "components": {
            "rag_chain":       chain_status,
            "fraud_predictor": predictor_status,
        },
        "missing_routers": _missing_routers or None,
        "endpoints":       endpoints,
        "docs":            "/docs" if settings.debug else None,
    })
