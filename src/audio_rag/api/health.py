"""Health check endpoints for Kubernetes probes."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

router = APIRouter()


class HealthStatus(BaseModel):
    """Health check response."""
    status: str
    checks: dict[str, bool] | None = None


@router.get("/live", response_model=HealthStatus)
async def liveness() -> HealthStatus:
    """Liveness probe - is the process running?
    
    Kubernetes uses this to know if it should restart the container.
    Always returns 200 if the process is alive.
    """
    return HealthStatus(status="alive")


@router.get("/ready")
async def readiness(request: Request) -> JSONResponse:
    """Readiness probe - can we accept traffic?
    
    Checks:
    - Redis connection (for queue)
    - Qdrant connection (for search)
    
    Returns 503 if any dependency is unhealthy.
    """
    checks = {}
    
    # Check Redis/Queue
    queue = getattr(request.app.state, "queue", None)
    if queue:
        try:
            checks["redis"] = queue.is_healthy()
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            checks["redis"] = False
    else:
        checks["redis"] = False
    
    # Check Qdrant (basic connectivity)
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333, timeout=2)
        client.get_collections()
        checks["qdrant"] = True
    except Exception as e:
        logger.warning(f"Qdrant health check failed: {e}")
        checks["qdrant"] = False
    
    all_healthy = all(checks.values())
    
    return JSONResponse(
        status_code=200 if all_healthy else 503,
        content={
            "status": "ready" if all_healthy else "not_ready",
            "checks": checks,
        },
    )


@router.get("/startup", response_model=HealthStatus)
async def startup(request: Request) -> JSONResponse:
    """Startup probe - has initialization completed?
    
    Kubernetes uses this to know when to start liveness/readiness checks.
    Returns 503 until the app is fully initialized.
    """
    initialized = getattr(request.app.state, "initialized", False)
    
    return JSONResponse(
        status_code=200 if initialized else 503,
        content={
            "status": "started" if initialized else "starting",
        },
    )
