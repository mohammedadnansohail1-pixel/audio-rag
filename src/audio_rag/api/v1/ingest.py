"""Audio ingestion endpoint."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Annotated

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status

from audio_rag.api.config import DEFAULT_API_CONFIG
from audio_rag.api.deps import (
    Context,
    Queue,
    get_request_context,
    get_queue,
    rate_limit_dependency,
)
from audio_rag.api.schemas import IngestResponse
from audio_rag.queue import IngestJob, Priority

logger = logging.getLogger(__name__)

router = APIRouter()


def map_priority(priority: str) -> Priority:
    """Map string priority to Priority enum."""
    return {
        "low": Priority.LOW,
        "normal": Priority.NORMAL,
        "high": Priority.HIGH,
    }.get(priority, Priority.NORMAL)


async def save_upload_file(
    file: UploadFile,
    max_size: int,
    allowed_extensions: set[str],
) -> Path:
    """Save uploaded file to temp location with validation.
    
    Args:
        file: Uploaded file
        max_size: Maximum file size in bytes
        allowed_extensions: Set of allowed extensions (e.g., {'.mp3', '.wav'})
    
    Returns:
        Path to saved file
        
    Raises:
        HTTPException: If validation fails
    """
    # Validate filename
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Filename required", "code": "INVALID_REQUEST"},
        )
    
    # Validate extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": f"Unsupported file format: {suffix}",
                "code": "INVALID_AUDIO",
                "allowed": list(allowed_extensions),
            },
        )
    
    # Create temp file with same extension
    temp_dir = Path(tempfile.gettempdir()) / "audio_rag_uploads"
    temp_dir.mkdir(exist_ok=True)
    
    temp_file = tempfile.NamedTemporaryFile(
        dir=temp_dir,
        suffix=suffix,
        delete=False,
    )
    temp_path = Path(temp_file.name)
    temp_file.close()
    
    # Stream file to disk with size check
    total_size = 0
    chunk_size = 1024 * 1024  # 1MB chunks
    
    try:
        async with aiofiles.open(temp_path, 'wb') as f:
            while chunk := await file.read(chunk_size):
                total_size += len(chunk)
                
                if total_size > max_size:
                    # Clean up and reject
                    temp_path.unlink(missing_ok=True)
                    max_mb = max_size / (1024 * 1024)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail={
                            "error": f"File too large (max {max_mb:.0f}MB)",
                            "code": "FILE_TOO_LARGE",
                            "max_bytes": max_size,
                        },
                    )
                
                await f.write(chunk)
        
        # Validate file is not empty
        if total_size == 0:
            temp_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"error": "Empty file", "code": "INVALID_AUDIO"},
            )
        
        logger.info(f"Saved upload: {temp_path} ({total_size} bytes)")
        return temp_path
        
    except HTTPException:
        raise
    except Exception as e:
        temp_path.unlink(missing_ok=True)
        logger.error(f"Failed to save upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to save file", "code": "INTERNAL_ERROR"},
        ) from e


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit audio for processing",
    dependencies=[Depends(rate_limit_dependency("ingest"))],
)
async def ingest_audio(
    request: Request,
    file: Annotated[UploadFile, File(description="Audio file to process")],
    ctx: Context,
    queue: Queue,
    collection: Annotated[str | None, Form()] = None,
    priority: Annotated[str, Form()] = "normal",
    metadata: Annotated[str | None, Form()] = None,  # JSON string
):
    """Submit an audio file for transcription and indexing.
    
    The file will be queued for processing. Use the returned job_id
    to check status via GET /api/v1/jobs/{job_id}.
    
    **Supported formats:** MP3, WAV, FLAC, OGG, M4A, AAC, OPUS, WEBM
    
    **Max file size:** 500MB
    """
    config = request.app.state.config or DEFAULT_API_CONFIG
    
    # Save uploaded file
    temp_path = await save_upload_file(
        file=file,
        max_size=config.upload.max_size_bytes,
        allowed_extensions=config.upload.allowed_extensions,
    )
    
    # Parse metadata if provided
    job_metadata = {}
    if metadata:
        import json
        try:
            job_metadata = json.loads(metadata)
        except json.JSONDecodeError:
            temp_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Invalid metadata JSON", "code": "INVALID_REQUEST"},
            )
    
    # Add original filename to metadata
    job_metadata["original_filename"] = file.filename
    job_metadata["uploaded_by"] = ctx.api_key_name
    
    # Use tenant's collection or provided one
    target_collection = collection or ctx.tenant_id
    
    # Create job
    job = IngestJob(
        tenant_id=target_collection,
        audio_path=str(temp_path),
        priority=map_priority(priority),
        metadata=job_metadata,
    )
    
    try:
        # Enqueue job
        job_id = queue.enqueue(job, validate=False)  # Already validated file
        
        # Get queue stats for wait estimate
        stats = queue.get_queue_stats()
        queue_name = {
            Priority.HIGH: "high",
            Priority.NORMAL: "normal",
            Priority.LOW: "low",
        }.get(job.priority, "normal")
        
        queue_depth = stats.get(queue_name, {}).get("depth", 0)
        # Rough estimate: 4 min per job ahead
        estimated_wait = queue_depth * 240 if queue_depth > 0 else None
        
        logger.info(
            f"Enqueued job {job_id} for {ctx.tenant_id} "
            f"(file={file.filename}, priority={priority})"
        )
        
        return IngestResponse(
            job_id=job_id,
            status="pending",
            collection=target_collection,
            filename=file.filename or "unknown",
            file_size_bytes=temp_path.stat().st_size,
            priority=priority,
            estimated_wait_seconds=estimated_wait,
        )
        
    except Exception as e:
        # Clean up temp file on failure
        temp_path.unlink(missing_ok=True)
        raise
