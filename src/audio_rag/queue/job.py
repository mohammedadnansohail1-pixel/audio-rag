"""Job dataclasses and enums for queue processing.

This module defines:
- Priority: Job priority levels
- JobStatus: Job lifecycle states
- JobStage: Processing pipeline stages
- IngestJob: Main job dataclass
- JobResult: Processing result
- JobCheckpoint: Recovery checkpoint
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any


class Priority(IntEnum):
    """Job priority levels.
    
    Higher value = higher priority.
    Workers process high priority jobs first.
    """
    
    LOW = 1       # Free tier, batch processing
    NORMAL = 2    # Standard paid tier
    HIGH = 3      # Priority processing add-on
    CRITICAL = 4  # System jobs, reprocessing


class JobStatus(StrEnum):
    """Job lifecycle states."""
    
    PENDING = "pending"       # In queue, waiting for worker
    RUNNING = "running"       # Worker processing
    COMPLETED = "completed"   # Successfully finished
    FAILED = "failed"         # Error during processing
    CANCELLED = "cancelled"   # Cancelled by user/system
    TIMEOUT = "timeout"       # Exceeded time limit


class JobStage(StrEnum):
    """Processing pipeline stages for checkpointing."""
    
    QUEUED = "queued"           # Job accepted, in queue
    VALIDATED = "validated"     # Pre-processing validation passed
    TRANSCRIBED = "transcribed" # ASR complete
    DIARIZED = "diarized"       # Speaker diarization complete
    ALIGNED = "aligned"         # Word-level alignment complete
    CHUNKED = "chunked"         # Text chunking complete
    EMBEDDED = "embedded"       # Embeddings generated
    STORED = "stored"           # Stored in vector DB
    COMPLETE = "complete"       # All processing done


# Stage order for recovery logic
STAGE_ORDER: list[JobStage] = [
    JobStage.QUEUED,
    JobStage.VALIDATED,
    JobStage.TRANSCRIBED,
    JobStage.DIARIZED,
    JobStage.ALIGNED,
    JobStage.CHUNKED,
    JobStage.EMBEDDED,
    JobStage.STORED,
    JobStage.COMPLETE,
]


def _utc_now() -> datetime:
    """Get current UTC time."""
    return datetime.now(timezone.utc)


def _generate_job_id() -> str:
    """Generate unique job ID."""
    return str(uuid.uuid4())


@dataclass
class IngestJob:
    """Audio ingestion job.
    
    Represents a request to process an audio file and store
    it in the vector database for a specific tenant.
    
    Attributes:
        job_id: Unique identifier (auto-generated if not provided)
        tenant_id: Tenant collection name (e.g., audio_rag_unt_cs_5500_fall2025)
        audio_path: Path to the audio file
        priority: Processing priority level
        created_at: Job creation timestamp
        config_overrides: Optional per-job configuration overrides
        metadata: Optional metadata to store with chunks
        callback_url: Optional webhook URL for completion notification
        idempotency_key: Optional key for deduplication (auto-generated from file hash)
    """
    
    tenant_id: str
    audio_path: str
    priority: Priority = Priority.NORMAL
    job_id: str = field(default_factory=_generate_job_id)
    created_at: datetime = field(default_factory=_utc_now)
    config_overrides: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    callback_url: str | None = None
    idempotency_key: str | None = None
    
    def __post_init__(self) -> None:
        """Validate and normalize fields after initialization."""
        # Ensure priority is Priority enum
        if isinstance(self.priority, int):
            self.priority = Priority(self.priority)
        
        # Ensure created_at is timezone-aware
        if self.created_at.tzinfo is None:
            self.created_at = self.created_at.replace(tzinfo=timezone.utc)
    
    def generate_idempotency_key(self) -> str:
        """Generate idempotency key from tenant_id and file hash.
        
        Returns:
            String key for deduplication: "{tenant_id}:{file_hash_prefix}"
        """
        if self.idempotency_key:
            return self.idempotency_key
        
        # Hash first 1MB of file for speed
        path = Path(self.audio_path)
        if path.exists():
            with open(path, "rb") as f:
                file_hash = hashlib.sha256(f.read(1024 * 1024)).hexdigest()[:16]
        else:
            # Fallback to path hash if file doesn't exist yet
            file_hash = hashlib.sha256(self.audio_path.encode()).hexdigest()[:16]
        
        self.idempotency_key = f"{self.tenant_id}:{file_hash}"
        return self.idempotency_key
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize job to dictionary for Redis storage."""
        return {
            "job_id": self.job_id,
            "tenant_id": self.tenant_id,
            "audio_path": self.audio_path,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "config_overrides": self.config_overrides,
            "metadata": self.metadata,
            "callback_url": self.callback_url,
            "idempotency_key": self.idempotency_key,
        }
    
    def to_json(self) -> str:
        """Serialize job to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IngestJob:
        """Deserialize job from dictionary."""
        return cls(
            job_id=data["job_id"],
            tenant_id=data["tenant_id"],
            audio_path=data["audio_path"],
            priority=Priority(data["priority"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            config_overrides=data.get("config_overrides", {}),
            metadata=data.get("metadata", {}),
            callback_url=data.get("callback_url"),
            idempotency_key=data.get("idempotency_key"),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> IngestJob:
        """Deserialize job from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class JobResult:
    """Result of a completed job.
    
    Attributes:
        job_id: The job this result belongs to
        status: Final job status
        stage: Last completed stage
        started_at: When processing started
        completed_at: When processing finished
        duration_seconds: Total processing time
        chunks_created: Number of chunks stored in vector DB
        error_message: Error details if failed
        error_type: Exception class name if failed
        output_path: Path to any output files
        metrics: Processing metrics (timing, memory, etc.)
    """
    
    job_id: str
    status: JobStatus
    stage: JobStage
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float | None = None
    chunks_created: int = 0
    error_message: str | None = None
    error_type: str | None = None
    output_path: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Calculate duration if not provided."""
        if (
            self.duration_seconds is None
            and self.started_at is not None
            and self.completed_at is not None
        ):
            self.duration_seconds = (
                self.completed_at - self.started_at
            ).total_seconds()
    
    @property
    def is_success(self) -> bool:
        """Check if job completed successfully."""
        return self.status == JobStatus.COMPLETED
    
    @property
    def is_terminal(self) -> bool:
        """Check if job is in a terminal state (won't change)."""
        return self.status in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.TIMEOUT,
        }
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "stage": self.stage.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "chunks_created": self.chunks_created,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "output_path": self.output_path,
            "metrics": self.metrics,
        }
    
    def to_json(self) -> str:
        """Serialize result to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobResult:
        """Deserialize result from dictionary."""
        return cls(
            job_id=data["job_id"],
            status=JobStatus(data["status"]),
            stage=JobStage(data["stage"]),
            started_at=(
                datetime.fromisoformat(data["started_at"])
                if data.get("started_at")
                else None
            ),
            completed_at=(
                datetime.fromisoformat(data["completed_at"])
                if data.get("completed_at")
                else None
            ),
            duration_seconds=data.get("duration_seconds"),
            chunks_created=data.get("chunks_created", 0),
            error_message=data.get("error_message"),
            error_type=data.get("error_type"),
            output_path=data.get("output_path"),
            metrics=data.get("metrics", {}),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> JobResult:
        """Deserialize result from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class JobCheckpoint:
    """Checkpoint for job recovery.
    
    Stores intermediate state to allow resuming failed jobs
    from the last successful stage.
    
    Attributes:
        job_id: The job this checkpoint belongs to
        stage: Last completed stage
        timestamp: When this checkpoint was created
        data: Stage-specific intermediate data
    """
    
    job_id: str
    stage: JobStage
    timestamp: datetime = field(default_factory=_utc_now)
    data: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize checkpoint to dictionary."""
        return {
            "job_id": self.job_id,
            "stage": self.stage.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
        }
    
    def to_json(self) -> str:
        """Serialize checkpoint to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobCheckpoint:
        """Deserialize checkpoint from dictionary."""
        return cls(
            job_id=data["job_id"],
            stage=JobStage(data["stage"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            data=data.get("data", {}),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> JobCheckpoint:
        """Deserialize checkpoint from JSON string."""
        return cls.from_dict(json.loads(json_str))


def get_next_stage(current: JobStage) -> JobStage | None:
    """Get the next stage in the pipeline.
    
    Args:
        current: Current stage
        
    Returns:
        Next stage, or None if current is the final stage
    """
    try:
        idx = STAGE_ORDER.index(current)
        if idx < len(STAGE_ORDER) - 1:
            return STAGE_ORDER[idx + 1]
        return None
    except ValueError:
        return None


def get_stage_index(stage: JobStage) -> int:
    """Get numeric index of a stage for comparison.
    
    Args:
        stage: The stage to look up
        
    Returns:
        Index in STAGE_ORDER (0-based)
        
    Raises:
        ValueError: If stage not found
    """
    return STAGE_ORDER.index(stage)
