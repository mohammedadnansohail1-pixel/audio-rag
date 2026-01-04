"""Queue-specific exceptions for Audio RAG.

Exception hierarchy:
    QueueError (base)
    ├── RedisConnectionError     - Redis unavailable
    ├── RedisOperationError      - Redis operation failed
    ├── QueueFullError           - Queue at max depth
    ├── JobValidationError       - Pre-queue validation failed
    │   ├── InvalidAudioError    - Audio file invalid
    │   └── TenantValidationError - Tenant invalid
    ├── DuplicateJobError        - Idempotency check failed
    ├── JobNotFoundError         - Job ID doesn't exist
    ├── JobTimeoutError          - Job exceeded timeout
    └── WorkerError (base)
        ├── WorkerResourceError  - Insufficient resources
        ├── WorkerStartupError   - Worker failed to start
        └── WorkerShutdownError  - Worker failed to shutdown cleanly
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class QueueError(Exception):
    """Base exception for all queue-related errors."""

    def __init__(self, message: str, *, recoverable: bool = True) -> None:
        """Initialize queue error.

        Args:
            message: Error description
            recoverable: Whether the operation can be retried
        """
        self.message = message
        self.recoverable = recoverable
        super().__init__(message)


# =============================================================================
# Redis Errors
# =============================================================================


class RedisConnectionError(QueueError):
    """Redis server is unavailable.

    Raised when:
    - Redis server is not running
    - Network connectivity issues
    - Authentication failure
    - Max retries exceeded
    """

    def __init__(
        self,
        message: str = "Redis connection failed",
        *,
        host: str | None = None,
        port: int | None = None,
        attempts: int = 0,
    ) -> None:
        self.host = host
        self.port = port
        self.attempts = attempts

        details = []
        if host:
            details.append(f"host={host}")
        if port:
            details.append(f"port={port}")
        if attempts:
            details.append(f"attempts={attempts}")

        full_message = message
        if details:
            full_message = f"{message} ({', '.join(details)})"

        super().__init__(full_message, recoverable=True)


class RedisOperationError(QueueError):
    """Redis operation failed after connection established.

    Raised when:
    - Command execution fails
    - Transaction fails
    - Lua script error
    """

    def __init__(
        self,
        message: str,
        *,
        operation: str | None = None,
        key: str | None = None,
    ) -> None:
        self.operation = operation
        self.key = key

        details = []
        if operation:
            details.append(f"operation={operation}")
        if key:
            details.append(f"key={key}")

        full_message = message
        if details:
            full_message = f"{message} ({', '.join(details)})"

        super().__init__(full_message, recoverable=True)


# =============================================================================
# Queue Errors
# =============================================================================


class QueueFullError(QueueError):
    """Queue has reached maximum depth.

    Raised when:
    - Queue depth exceeds configured limit
    - System is overloaded
    - Backpressure mechanism triggered
    """

    def __init__(
        self,
        message: str = "Queue is full",
        *,
        queue_name: str | None = None,
        current_depth: int | None = None,
        max_depth: int | None = None,
    ) -> None:
        self.queue_name = queue_name
        self.current_depth = current_depth
        self.max_depth = max_depth

        details = []
        if queue_name:
            details.append(f"queue={queue_name}")
        if current_depth is not None and max_depth is not None:
            details.append(f"depth={current_depth}/{max_depth}")

        full_message = message
        if details:
            full_message = f"{message} ({', '.join(details)})"

        # Not immediately recoverable - need to wait for queue to drain
        super().__init__(full_message, recoverable=False)


class DuplicateJobError(QueueError):
    """Job already exists (idempotency check failed).

    Raised when:
    - Same file uploaded twice for same tenant
    - Job with same idempotency key already queued/processing
    """

    def __init__(
        self,
        existing_job_id: str,
        *,
        idempotency_key: str | None = None,
    ) -> None:
        self.existing_job_id = existing_job_id
        self.idempotency_key = idempotency_key

        message = f"Duplicate job detected, existing job_id: {existing_job_id}"
        if idempotency_key:
            message = f"{message} (key={idempotency_key})"

        # Not an error per se - return existing job_id to client
        super().__init__(message, recoverable=False)


class JobNotFoundError(QueueError):
    """Job ID does not exist.

    Raised when:
    - Querying status of non-existent job
    - Job expired from Redis
    - Invalid job_id format
    """

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(f"Job not found: {job_id}", recoverable=False)


class JobTimeoutError(QueueError):
    """Job exceeded maximum processing time.

    Raised when:
    - Worker timeout exceeded
    - Processing took longer than allowed
    """

    def __init__(
        self,
        job_id: str,
        *,
        timeout_seconds: int | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        self.job_id = job_id
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds

        message = f"Job timed out: {job_id}"
        if timeout_seconds:
            message = f"{message} (timeout={timeout_seconds}s"
            if elapsed_seconds:
                message = f"{message}, elapsed={elapsed_seconds:.1f}s)"
            else:
                message = f"{message})"

        super().__init__(message, recoverable=True)


# =============================================================================
# Validation Errors
# =============================================================================


class JobValidationError(QueueError):
    """Job failed pre-queue validation.

    Base class for validation errors. Raised when:
    - Required fields missing
    - Field values invalid
    """

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        value: str | None = None,
    ) -> None:
        self.field = field
        self.value = value

        full_message = message
        if field:
            full_message = f"{message} (field={field}"
            if value:
                full_message = f"{full_message}, value={value})"
            else:
                full_message = f"{full_message})"

        super().__init__(full_message, recoverable=False)


class InvalidAudioError(JobValidationError):
    """Audio file is invalid or unsupported.

    Raised when:
    - File doesn't exist
    - File is corrupted
    - Unsupported format
    - File too large
    - Duration exceeds limit
    - File is silent
    """

    def __init__(
        self,
        message: str,
        *,
        audio_path: Path | str | None = None,
        reason: str | None = None,
    ) -> None:
        self.audio_path = audio_path
        self.reason = reason

        full_message = message
        details = []
        if audio_path:
            details.append(f"path={audio_path}")
        if reason:
            details.append(f"reason={reason}")
        if details:
            full_message = f"{message} ({', '.join(details)})"

        super().__init__(full_message, field="audio_path")


class TenantValidationError(JobValidationError):
    """Tenant ID is invalid or unknown.

    Raised when:
    - Tenant ID format invalid
    - Tenant doesn't exist in system
    - Tenant is disabled/suspended
    """

    def __init__(
        self,
        message: str,
        *,
        tenant_id: str | None = None,
        reason: str | None = None,
    ) -> None:
        self.tenant_id = tenant_id
        self.reason = reason

        full_message = message
        details = []
        if tenant_id:
            details.append(f"tenant_id={tenant_id}")
        if reason:
            details.append(f"reason={reason}")
        if details:
            full_message = f"{message} ({', '.join(details)})"

        super().__init__(full_message, field="tenant_id")


# =============================================================================
# Worker Errors
# =============================================================================


class WorkerError(QueueError):
    """Base exception for worker-related errors."""

    def __init__(
        self,
        message: str,
        *,
        worker_id: str | None = None,
        recoverable: bool = True,
    ) -> None:
        self.worker_id = worker_id

        full_message = message
        if worker_id:
            full_message = f"{message} (worker={worker_id})"

        super().__init__(full_message, recoverable=recoverable)


class WorkerResourceError(WorkerError):
    """Worker lacks resources to process job.

    Raised when:
    - Insufficient GPU memory
    - Insufficient disk space
    - CPU overloaded
    """

    def __init__(
        self,
        message: str,
        *,
        worker_id: str | None = None,
        resource: str | None = None,
        required: float | None = None,
        available: float | None = None,
        unit: str = "bytes",
    ) -> None:
        self.resource = resource
        self.required = required
        self.available = available
        self.unit = unit

        details = []
        if resource:
            details.append(f"resource={resource}")
        if required is not None and available is not None:
            details.append(f"required={required}{unit}, available={available}{unit}")

        full_message = message
        if details:
            full_message = f"{message} ({', '.join(details)})"

        super().__init__(full_message, worker_id=worker_id, recoverable=True)


class WorkerStartupError(WorkerError):
    """Worker failed to start.

    Raised when:
    - Model loading failed
    - GPU initialization failed
    - Configuration invalid
    """

    def __init__(
        self,
        message: str,
        *,
        worker_id: str | None = None,
        stage: str | None = None,
    ) -> None:
        self.stage = stage

        full_message = message
        if stage:
            full_message = f"{message} (stage={stage})"

        super().__init__(full_message, worker_id=worker_id, recoverable=False)


class WorkerShutdownError(WorkerError):
    """Worker failed to shutdown cleanly.

    Raised when:
    - Timeout during shutdown
    - Resources not released
    - Job left in inconsistent state
    """

    def __init__(
        self,
        message: str,
        *,
        worker_id: str | None = None,
        pending_jobs: int | None = None,
    ) -> None:
        self.pending_jobs = pending_jobs

        full_message = message
        if pending_jobs:
            full_message = f"{message} (pending_jobs={pending_jobs})"

        super().__init__(full_message, worker_id=worker_id, recoverable=False)
