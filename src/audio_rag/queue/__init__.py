"""Job queue module for async audio processing."""

from audio_rag.queue.job import (
    Priority,
    JobStatus,
    JobStage,
    STAGE_ORDER,
    IngestJob,
    JobResult,
    JobCheckpoint,
    get_next_stage,
    get_stage_index,
)
from audio_rag.queue.exceptions import (
    QueueError,
    RedisConnectionError,
    RedisOperationError,
    QueueFullError,
    DuplicateJobError,
    JobNotFoundError,
    JobTimeoutError,
    JobValidationError,
    InvalidAudioError,
    TenantValidationError,
    WorkerError,
    WorkerResourceError,
    WorkerStartupError,
    WorkerShutdownError,
)
from audio_rag.queue.queue import AudioRAGQueue
from audio_rag.queue.config import QueueConfig, DEFAULT_QUEUE_CONFIG

__all__ = [
    # Job types
    "Priority",
    "JobStatus",
    "JobStage",
    "STAGE_ORDER",
    "IngestJob",
    "JobResult",
    "JobCheckpoint",
    "get_next_stage",
    "get_stage_index",
    # Exceptions
    "QueueError",
    "RedisConnectionError",
    "RedisOperationError",
    "QueueFullError",
    "DuplicateJobError",
    "JobNotFoundError",
    "JobTimeoutError",
    "JobValidationError",
    "InvalidAudioError",
    "TenantValidationError",
    "WorkerError",
    "WorkerResourceError",
    "WorkerStartupError",
    "WorkerShutdownError",
    # Queue
    "AudioRAGQueue",
    "QueueConfig",
    "DEFAULT_QUEUE_CONFIG",
]
