"""
Health check endpoints for the FinSight API.

Why health endpoints matter in production
------------------------------------------
Production infrastructure treats your service as a black box. The only way it
knows whether a process is working is by asking it — and that question is the
health endpoint. Without one, three critical systems break down:

  KUBERNETES / ECS ORCHESTRATION
  Container orchestrators run two probes on every pod:
    • Liveness probe  — "Is the process alive?"
      A failing liveness check causes the orchestrator to kill and restart
      the container. Without it, a deadlocked or OOM-killed process that
      hasn't exited stays in the load-balancer rotation silently returning
      500s forever.
    • Readiness probe — "Is the service ready to handle traffic?"
      A failing readiness check removes the pod from the load-balancer pool
      without restarting it. This is how rolling deployments work safely:
      new pods only receive traffic after they pass readiness (i.e. after the
      ML models have finished loading), preventing requests from hitting a pod
      that is still warming up.

  LOAD BALANCER ROUTING
  AWS ALB, nginx, and HAProxy all support health-check-based routing. A
  target that fails its health check is drained from the pool and no new
  connections are sent to it. Without a health endpoint the load balancer
  cannot distinguish a healthy pod from a crashed one.

  ON-CALL ALERTING
  Uptime monitors (Datadog, PagerDuty, Pingdom) hit the health endpoint on a
  fixed cadence (typically every 30s). A non-200 response triggers a page.
  Without it, the on-call engineer only finds out about an outage when a user
  reports it.

Endpoint map (prefix /health from main.py)
------------------------------------------
  GET /health        — full status: versions, component health, uptime
  GET /health/live   — liveness:  200 if process is running, 503 otherwise
  GET /health/ready  — readiness: 200 if models are loaded, 503 otherwise
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from api.config import get_settings

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# FULL STATUS
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "",
    summary     = "Full service health status",
    description = "Returns version, uptime, and the loaded state of each ML component.",
)
async def health(request: Request) -> JSONResponse:
    """
    Full health snapshot — suitable for dashboards and human inspection.

    Returns HTTP 200 with status="degraded" when some components failed to
    load (the process is alive but not fully functional), so monitoring tools
    that alert on non-200 responses do not false-page for partial degradation.
    Use GET /health/ready for a strict binary probe.
    """
    settings   = get_settings()
    started_at = getattr(request.app.state, "started_at", None)

    uptime_seconds: float | None = None
    if started_at:
        try:
            start_ts      = datetime.strptime(started_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            uptime_seconds = round((datetime.now(timezone.utc) - start_ts).total_seconds(), 1)
        except ValueError:
            pass

    chain_loaded     = (
        getattr(request.app.state, "chain", None) is not None
    )
    predictor_loaded = (
        getattr(request.app.state, "predictor", None) is not None
    )

    # Gather component-level detail without crashing if a component is None.
    chain_detail: dict = {"loaded": chain_loaded}
    if chain_loaded:
        chain = request.app.state.chain
        chain_detail.update({
            "vector_count": chain.embedder.vector_count,
            "bedrock_stub": chain.bedrock._stub_mode,
        })

    predictor_detail: dict = {"loaded": predictor_loaded}
    if predictor_loaded:
        p = request.app.state.predictor
        predictor_detail.update({
            "features":   len(p.feature_cols),
            "threshold":  round(p.threshold, 4),
            "store_size": p.store_size(),
        })

    all_healthy = chain_loaded and predictor_loaded
    status      = "healthy" if all_healthy else "degraded"

    return JSONResponse(
        status_code = 200,   # always 200 here — use /ready for a strict probe
        content     = {
            "status":      status,
            "service":     "FinSight AI Platform",
            "version":     request.app.version,
            "environment": settings.env,
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "started_at":  started_at,
            "uptime_seconds": uptime_seconds,
            "components": {
                "rag_chain":       chain_detail,
                "fraud_predictor": predictor_detail,
            },
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# LIVENESS PROBE
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/live",
    summary     = "Liveness probe",
    description = "Returns 200 if the process is running. Used by Kubernetes livenessProbe.",
)
async def liveness() -> JSONResponse:
    """
    Minimal liveness probe — answers 'is the process alive?'

    Intentionally does NO work beyond returning a response. Liveness probes
    should never fail due to a downstream dependency (database down, model not
    loaded) — only a truly hung or deadlocked process should fail liveness,
    because failure triggers a pod restart which is disruptive and should be
    a last resort.

    Kubernetes livenessProbe config example:
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 15
          failureThreshold: 3
    """
    return JSONResponse(
        status_code = 200,
        content     = {
            "status":    "alive",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# READINESS PROBE
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/ready",
    summary     = "Readiness probe",
    description = "Returns 200 only when all ML components are loaded. Used by Kubernetes readinessProbe.",
)
async def readiness(request: Request) -> JSONResponse:
    """
    Strict readiness probe — answers 'is the service ready to serve traffic?'

    Returns 503 if any required component failed to load during startup.
    This prevents the load balancer from routing requests to a pod that is
    still warming up or that started with a broken artifact path.

    Unlike the liveness probe, readiness SHOULD fail when dependencies are
    not yet available — that is its entire purpose. The load balancer removes
    the pod from rotation on 503, and puts it back when 200 is returned.

    Kubernetes readinessProbe config example:
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 20   # models take ~5s to load; give headroom
          periodSeconds: 10
          failureThreshold: 2
    """
    chain_ready     = getattr(request.app.state, "chain", None) is not None
    predictor_ready = getattr(request.app.state, "predictor", None) is not None
    is_ready        = chain_ready and predictor_ready

    not_ready = []
    if not chain_ready:
        not_ready.append("rag_chain")
    if not predictor_ready:
        not_ready.append("fraud_predictor")

    return JSONResponse(
        status_code = 200 if is_ready else 503,
        content     = {
            "status":        "ready" if is_ready else "not_ready",
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "not_ready":     not_ready or None,
        },
    )
