"""Request and response schemas for API endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


# ============================================================================
# Common Schemas
# ============================================================================

class ErrorResponse(BaseModel):
    """Standard error response format."""
    
    error: str = Field(description="Human-readable error message")
    code: str = Field(description="Machine-readable error code")
    status: int = Field(description="HTTP status code")
    request_id: str | None = Field(default=None, description="Request ID for tracing")
    details: dict[str, Any] | None = Field(default=None, description="Additional context")
    retry_after: int | None = Field(default=None, description="Seconds until retry allowed")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "Rate limit exceeded",
                "code": "RATE_LIMITED",
                "status": 429,
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "retry_after": 60,
            }
        }
    }


class PaginationParams(BaseModel):
    """Pagination parameters."""
    
    limit: int = Field(default=10, ge=1, le=100, description="Max results to return")
    offset: int = Field(default=0, ge=0, description="Number of results to skip")


# ============================================================================
# Ingest Schemas
# ============================================================================

class IngestConfig(BaseModel):
    """Configuration overrides for ingestion."""
    
    language: str | None = Field(default=None, description="Force language (e.g., 'en', 'es')")
    enable_diarization: bool = Field(default=True, description="Enable speaker diarization")
    max_speakers: int | None = Field(default=None, ge=1, le=20, description="Max speakers to detect")
    chunk_duration_seconds: float = Field(default=30.0, ge=5, le=300, description="Target chunk duration")


class IngestRequest(BaseModel):
    """Request body for audio ingestion (metadata only, file is multipart)."""
    
    collection: str | None = Field(
        default=None,
        description="Collection name (defaults to tenant's default collection)",
        min_length=5,
        max_length=128,
        pattern=r"^[a-z0-9_]+$",
    )
    priority: Literal["low", "normal", "high"] = Field(
        default="normal",
        description="Processing priority",
    )
    config: IngestConfig | None = Field(
        default=None,
        description="Processing configuration overrides",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Custom metadata to attach to chunks",
    )
    callback_url: HttpUrl | None = Field(
        default=None,
        description="Webhook URL to notify on completion",
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "collection": "audio_rag_unt_cs_5500_fall2025",
                "priority": "normal",
                "metadata": {"lecture": "Week 1 - Introduction", "professor": "Dr. Smith"},
            }
        }
    }


class IngestResponse(BaseModel):
    """Response after successful ingest submission."""
    
    job_id: str = Field(description="Unique job identifier")
    status: str = Field(description="Current job status")
    collection: str = Field(description="Target collection name")
    filename: str = Field(description="Original filename")
    file_size_bytes: int = Field(description="File size in bytes")
    priority: str = Field(description="Processing priority")
    estimated_wait_seconds: int | None = Field(
        default=None,
        description="Estimated wait time before processing starts",
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "pending",
                "collection": "audio_rag_unt_cs_5500_fall2025",
                "filename": "lecture_week1.mp3",
                "file_size_bytes": 52428800,
                "priority": "normal",
                "estimated_wait_seconds": 120,
            }
        }
    }


# ============================================================================
# Job Schemas
# ============================================================================

class JobStatusResponse(BaseModel):
    """Job status information."""
    
    job_id: str = Field(description="Unique job identifier")
    status: Literal["pending", "running", "completed", "failed", "cancelled", "timeout"] = Field(
        description="Current job status",
    )
    stage: str = Field(description="Current processing stage")
    progress: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Progress (0.0 to 1.0)",
    )
    created_at: datetime = Field(description="Job creation time")
    started_at: datetime | None = Field(default=None, description="Processing start time")
    completed_at: datetime | None = Field(default=None, description="Completion time")
    duration_seconds: float | None = Field(default=None, description="Total processing duration")
    error: str | None = Field(default=None, description="Error message if failed")
    result: dict[str, Any] | None = Field(default=None, description="Result data if completed")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "running",
                "stage": "transcribed",
                "progress": 0.6,
                "created_at": "2025-01-03T12:00:00Z",
                "started_at": "2025-01-03T12:00:05Z",
            }
        }
    }


class JobCancelResponse(BaseModel):
    """Response after job cancellation."""
    
    job_id: str
    cancelled: bool
    message: str


# ============================================================================
# Query Schemas
# ============================================================================

class QueryRequest(BaseModel):
    """Search query request."""
    
    query: str = Field(
        description="Search query text",
        min_length=1,
        max_length=2000,
    )
    collection: str | None = Field(
        default=None,
        description="Collection to search (defaults to tenant's default)",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum results to return",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Number of results to skip",
    )
    score_threshold: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Minimum similarity score",
    )
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Metadata filters",
    )
    include_context: bool = Field(
        default=True,
        description="Include surrounding context in results",
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What are the main topics covered in the introduction?",
                "collection": "audio_rag_unt_cs_5500_fall2025",
                "limit": 5,
                "filters": {"lecture": "Week 1 - Introduction"},
            }
        }
    }


class SearchResult(BaseModel):
    """Single search result."""
    
    chunk_id: str = Field(description="Unique chunk identifier")
    text: str = Field(description="Chunk text content")
    score: float = Field(description="Similarity score (0-1)")
    start_time: float | None = Field(default=None, description="Start time in audio (seconds)")
    end_time: float | None = Field(default=None, description="End time in audio (seconds)")
    speaker: str | None = Field(default=None, description="Speaker label if diarized")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    context_before: str | None = Field(default=None, description="Text before this chunk")
    context_after: str | None = Field(default=None, description="Text after this chunk")


class QueryResponse(BaseModel):
    """Search query response."""
    
    results: list[SearchResult] = Field(description="Search results")
    total: int = Field(description="Total matching results (before limit)")
    query: str = Field(description="Original query")
    collection: str = Field(description="Collection searched")
    query_time_ms: float = Field(description="Query execution time in milliseconds")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "results": [
                    {
                        "chunk_id": "chunk-001",
                        "text": "Today we'll cover the fundamentals of data engineering...",
                        "score": 0.89,
                        "start_time": 45.2,
                        "end_time": 75.8,
                        "speaker": "SPEAKER_00",
                        "metadata": {"lecture": "Week 1"},
                    }
                ],
                "total": 15,
                "query": "What are the main topics?",
                "collection": "audio_rag_unt_cs_5500_fall2025",
                "query_time_ms": 45.2,
            }
        }
    }


# ============================================================================
# Collection Schemas
# ============================================================================

class CollectionInfo(BaseModel):
    """Collection information."""
    
    name: str = Field(description="Collection name")
    chunks_count: int = Field(description="Number of chunks in collection")
    created_at: datetime | None = Field(default=None, description="Creation time")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Collection metadata")


class CollectionListResponse(BaseModel):
    """List of collections."""
    
    collections: list[CollectionInfo]
    total: int
