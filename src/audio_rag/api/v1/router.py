"""API v1 router aggregator."""
from fastapi import APIRouter

from audio_rag.api.v1.ingest import router as ingest_router
from audio_rag.api.v1.jobs import router as jobs_router
from audio_rag.api.v1.query import router as query_router
from audio_rag.api.v1.streaming import router as streaming_router
from audio_rag.api.v1.collections import router as collections_router

router = APIRouter()

router.include_router(collections_router, tags=["collections"])
router.include_router(ingest_router, tags=["ingest"])
router.include_router(jobs_router, tags=["jobs"])
router.include_router(query_router, tags=["query"])
router.include_router(streaming_router, tags=["streaming"])


@router.get("/", summary="API Information")
async def api_info():
    """Get API version and status information."""
    return {
        "version": "v1",
        "status": "active",
        "endpoints": {
            "collections": "/api/v1/collections",
            "ingest": "/api/v1/ingest",
            "jobs": "/api/v1/jobs/{job_id}",
            "query": "/api/v1/query",
            "streaming": "/api/v1/ws/transcribe",
        },
    }
