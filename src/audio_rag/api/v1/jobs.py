"""Job status and management endpoints."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from audio_rag.api.deps import (
    Context,
    Queue,
    get_request_context,
    get_queue,
    rate_limit_dependency,
)
from audio_rag.api.schemas import JobCancelResponse, JobStatusResponse
from audio_rag.queue import JobNotFoundError, JobStatus, STAGE_ORDER

logger = logging.getLogger(__name__)

router = APIRouter()


def calculate_progress(stage: str, job_status: str) -> float | None:
    """Calculate progress percentage from stage."""
    if job_status == "completed":
        return 1.0
    if job_status in ("failed", "cancelled", "timeout"):
        return None
    if job_status == "pending":
        return 0.0
    
    try:
        stage_idx = STAGE_ORDER.index(stage.lower())
        return round(stage_idx / len(STAGE_ORDER), 2)
    except ValueError:
        return None


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Get job status",
    dependencies=[Depends(rate_limit_dependency("status"))],
)
async def get_job_status(
    job_id: str,
    ctx: Context,
    queue: Queue,
):
    """Get the current status of a processing job."""
    try:
        job_status = queue.get_status(job_id)
        
        if not job_status:
            raise JobNotFoundError(job_id)
        
        job_data = queue.get_job(job_id)
        created_at = datetime.now(timezone.utc)
        if job_data:
            created_at = job_data.created_at
        
        result = None
        error = None
        started_at = None
        completed_at = None
        duration = None
        
        job_result = queue.get_result(job_id)
        if job_result:
            started_at = job_result.started_at
            completed_at = job_result.completed_at
            duration = job_result.duration_seconds
            
            if job_result.status == JobStatus.COMPLETED:
                result = {
                    "chunks_created": job_result.chunks_created,
                    "output_path": job_result.output_path,
                    "metrics": job_result.metrics,
                }
            elif job_result.status in (JobStatus.FAILED, JobStatus.TIMEOUT):
                error = job_result.error_message
        
        status_map = {
            "pending": "pending",
            "queued": "pending",
            "running": "running",
            "completed": "completed",
            "failed": "failed",
            "cancelled": "cancelled",
            "timeout": "timeout",
        }
        
        stage = job_status.get("stage", "queued")
        current_status = status_map.get(job_status.get("status", "pending"), "pending")
        
        return JobStatusResponse(
            job_id=job_id,
            status=current_status,
            stage=stage,
            progress=calculate_progress(stage, current_status),
            created_at=created_at,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
            error=error,
            result=result,
        )
        
    except JobNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": f"Job not found: {job_id}", "code": "JOB_NOT_FOUND"},
        )


@router.delete(
    "/jobs/{job_id}",
    response_model=JobCancelResponse,
    summary="Cancel a job",
    dependencies=[Depends(rate_limit_dependency("status"))],
)
async def cancel_job(
    job_id: str,
    ctx: Context,
    queue: Queue,
):
    """Cancel a pending job."""
    try:
        cancelled = queue.cancel(job_id)
        
        if cancelled:
            logger.info(f"Cancelled job {job_id} by {ctx.tenant_id}")
            return JobCancelResponse(
                job_id=job_id,
                cancelled=True,
                message="Job cancelled successfully",
            )
        else:
            return JobCancelResponse(
                job_id=job_id,
                cancelled=False,
                message="Job cannot be cancelled (already running or completed)",
            )
            
    except JobNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": f"Job not found: {job_id}", "code": "JOB_NOT_FOUND"},
        )
