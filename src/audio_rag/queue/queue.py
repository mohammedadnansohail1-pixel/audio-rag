"""Redis-backed job queue with RQ.

Provides:
- Priority queue support (high, normal, low)
- Idempotency (prevent duplicate jobs)
- Queue depth limits (backpressure)
- Job status tracking
- Tenant-aware queuing
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from rq import Queue
from rq.job import Job as RQJob

from audio_rag.queue.config import QueueConfig
from audio_rag.queue.connection import RedisConnectionManager
from audio_rag.queue.exceptions import (
    DuplicateJobError,
    JobNotFoundError,
    QueueFullError,
    RedisConnectionError,
)
from audio_rag.queue.job import (
    IngestJob,
    JobResult,
    JobStage,
    JobStatus,
    Priority,
)
from audio_rag.queue.validation import JobValidator

if TYPE_CHECKING:
    from redis import Redis

logger = logging.getLogger(__name__)


# Redis key prefixes
KEY_PREFIX = "audio_rag:"
IDEMPOTENCY_PREFIX = f"{KEY_PREFIX}idempotency:"
JOB_STATUS_PREFIX = f"{KEY_PREFIX}job_status:"
JOB_DATA_PREFIX = f"{KEY_PREFIX}job_data:"
QUEUE_STATS_PREFIX = f"{KEY_PREFIX}queue_stats:"


class AudioRAGQueue:
    """Main queue interface for Audio RAG jobs.
    
    Features:
    - Priority-based queuing (high > normal > low)
    - Idempotency checking (prevent duplicate processing)
    - Queue depth limits with backpressure
    - Job status tracking
    - Pre-queue validation
    
    Example:
        queue = AudioRAGQueue.from_config(config)
        
        job = IngestJob(
            tenant_id="audio_rag_unt_cs_5500_fall2025",
            audio_path="/uploads/lecture.mp3",
        )
        
        job_id = queue.enqueue(job)
        status = queue.get_status(job_id)
    """
    
    def __init__(
        self,
        connection_manager: RedisConnectionManager,
        config: QueueConfig,
        validator: JobValidator | None = None,
    ) -> None:
        """Initialize queue.
        
        Args:
            connection_manager: Redis connection manager
            config: Queue configuration
            validator: Job validator (creates default if not provided)
        """
        self.connection_manager = connection_manager
        self.config = config
        self.validator = validator or JobValidator()
        
        # Create RQ queues for each priority level
        self._queues: dict[str, Queue] = {}
        self._init_queues()
    
    @classmethod
    def from_config(cls, config: QueueConfig | None = None) -> AudioRAGQueue:
        """Create queue from configuration.
        
        Args:
            config: Queue configuration (uses defaults if not provided)
        """
        config = config or QueueConfig()
        connection_manager = RedisConnectionManager(config.redis)
        return cls(connection_manager, config)
    
    def _init_queues(self) -> None:
        """Initialize RQ queues."""
        conn = self.connection_manager.get_connection()
        
        for queue_def in self.config.queues:
            self._queues[queue_def.name] = Queue(
                name=queue_def.name,
                connection=conn,
                default_timeout=queue_def.timeout,
            )
            logger.debug(f"Initialized queue: {queue_def.name}")
    
    def _get_queue_for_priority(self, priority: Priority) -> Queue:
        """Get the appropriate queue for a priority level."""
        if priority >= Priority.HIGH:
            queue_name = "high"
        elif priority == Priority.NORMAL:
            queue_name = "normal"
        else:
            queue_name = "low"
        
        # Fall back to first available queue if not found
        if queue_name not in self._queues:
            queue_name = list(self._queues.keys())[0]
        
        return self._queues[queue_name]
    
    def _get_redis(self) -> Redis:
        """Get Redis connection."""
        return self.connection_manager.get_connection()
    
    def _check_idempotency(self, job: IngestJob) -> str | None:
        """Check if job already exists.
        
        Returns:
            Existing job_id if duplicate, None if new
        """
        key = job.generate_idempotency_key()
        redis_key = f"{IDEMPOTENCY_PREFIX}{key}"
        
        existing = self._get_redis().get(redis_key)
        if existing:
            return existing  # Returns job_id
        
        return None
    
    def _set_idempotency(self, job: IngestJob) -> None:
        """Set idempotency key for job."""
        key = job.generate_idempotency_key()
        redis_key = f"{IDEMPOTENCY_PREFIX}{key}"
        
        self._get_redis().setex(
            redis_key,
            self.config.idempotency_ttl,
            job.job_id,
        )
    
    def _check_queue_depth(self, queue: Queue) -> None:
        """Check if queue has room for more jobs.
        
        Raises:
            QueueFullError: If queue is at max depth
        """
        queue_def = self.config.get_queue(queue.name)
        if queue_def is None:
            return
        
        current_depth = len(queue)
        if current_depth >= queue_def.max_depth:
            raise QueueFullError(
                "Queue is at capacity",
                queue_name=queue.name,
                current_depth=current_depth,
                max_depth=queue_def.max_depth,
            )
    
    def _store_job_data(self, job: IngestJob) -> None:
        """Store job data in Redis."""
        redis_key = f"{JOB_DATA_PREFIX}{job.job_id}"
        self._get_redis().setex(
            redis_key,
            self.config.result_ttl,
            job.to_json(),
        )
    
    def _store_job_status(
        self,
        job_id: str,
        status: JobStatus,
        stage: JobStage,
        error: str | None = None,
    ) -> None:
        """Store job status in Redis."""
        redis_key = f"{JOB_STATUS_PREFIX}{job_id}"
        
        data = {
            "status": status.value,
            "stage": stage.value,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "error": error,
        }
        
        self._get_redis().setex(
            redis_key,
            self.config.result_ttl,
            json.dumps(data),
        )
    
    def enqueue(
        self,
        job: IngestJob,
        *,
        validate: bool = True,
        check_idempotency: bool = True,
    ) -> str:
        """Add a job to the queue.
        
        Args:
            job: The job to enqueue
            validate: Whether to validate the job before queuing
            check_idempotency: Whether to check for duplicate jobs
            
        Returns:
            The job ID
            
        Raises:
            InvalidAudioError: If audio validation fails
            TenantValidationError: If tenant validation fails
            DuplicateJobError: If job already exists
            QueueFullError: If queue is at capacity
            RedisConnectionError: If Redis is unavailable
        """
        # Validate job
        if validate:
            self.validator.validate(job)
        
        # Check idempotency
        if check_idempotency:
            existing_id = self._check_idempotency(job)
            if existing_id:
                raise DuplicateJobError(
                    existing_id,
                    idempotency_key=job.idempotency_key,
                )
        
        # Get queue and check depth
        queue = self._get_queue_for_priority(job.priority)
        self._check_queue_depth(queue)
        
        # Store job data and set idempotency
        self._store_job_data(job)
        if check_idempotency:
            self._set_idempotency(job)
        
        # Set initial status
        self._store_job_status(job.job_id, JobStatus.PENDING, JobStage.QUEUED)
        
        # Enqueue to RQ
        # The actual processing function will be defined in worker.py
        rq_job = queue.enqueue(
            "audio_rag.queue.worker.process_ingest_job",
            job.to_dict(),
            job_id=job.job_id,
            job_timeout=self.config.get_timeout_for_queue(queue.name),
            result_ttl=self.config.result_ttl,
            failure_ttl=self.config.result_ttl,
        )
        
        logger.info(
            f"Enqueued job {job.job_id} to queue '{queue.name}' "
            f"(tenant={job.tenant_id}, priority={job.priority.name})"
        )
        
        return job.job_id
    
    def get_status(self, job_id: str) -> dict:
        """Get job status.
        
        Args:
            job_id: The job ID
            
        Returns:
            Dict with status, stage, updated_at, error
            
        Raises:
            JobNotFoundError: If job doesn't exist
        """
        redis_key = f"{JOB_STATUS_PREFIX}{job_id}"
        data = self._get_redis().get(redis_key)
        
        if not data:
            raise JobNotFoundError(job_id)
        
        return json.loads(data)
    
    def get_job(self, job_id: str) -> IngestJob:
        """Get job data.
        
        Args:
            job_id: The job ID
            
        Returns:
            The IngestJob
            
        Raises:
            JobNotFoundError: If job doesn't exist
        """
        redis_key = f"{JOB_DATA_PREFIX}{job_id}"
        data = self._get_redis().get(redis_key)
        
        if not data:
            raise JobNotFoundError(job_id)
        
        return IngestJob.from_json(data)
    
    def get_result(self, job_id: str) -> JobResult | None:
        """Get job result if completed.
        
        Args:
            job_id: The job ID
            
        Returns:
            JobResult if completed, None if still processing
            
        Raises:
            JobNotFoundError: If job doesn't exist
        """
        status = self.get_status(job_id)
        
        if status["status"] not in {
            JobStatus.COMPLETED.value,
            JobStatus.FAILED.value,
            JobStatus.TIMEOUT.value,
        }:
            return None
        
        # Get result from RQ job
        for queue in self._queues.values():
            try:
                rq_job = RQJob.fetch(job_id, connection=self._get_redis())
                if rq_job.result:
                    return JobResult.from_dict(rq_job.result)
            except Exception:
                continue
        
        # Build result from status
        return JobResult(
            job_id=job_id,
            status=JobStatus(status["status"]),
            stage=JobStage(status["stage"]),
            error_message=status.get("error"),
        )
    
    def cancel(self, job_id: str) -> bool:
        """Cancel a pending job.
        
        Args:
            job_id: The job ID
            
        Returns:
            True if cancelled, False if already processing/completed
        """
        status = self.get_status(job_id)
        
        if status["status"] != JobStatus.PENDING.value:
            return False
        
        # Try to remove from RQ
        try:
            rq_job = RQJob.fetch(job_id, connection=self._get_redis())
            rq_job.cancel()
        except Exception:
            pass
        
        # Update status
        self._store_job_status(
            job_id,
            JobStatus.CANCELLED,
            JobStage(status["stage"]),
        )
        
        logger.info(f"Cancelled job {job_id}")
        return True
    
    def get_queue_stats(self) -> dict[str, dict]:
        """Get statistics for all queues.
        
        Returns:
            Dict mapping queue name to stats dict
        """
        stats = {}
        
        for name, queue in self._queues.items():
            queue_def = self.config.get_queue(name)
            max_depth = queue_def.max_depth if queue_def else 100
            
            stats[name] = {
                "depth": len(queue),
                "max_depth": max_depth,
                "utilization": len(queue) / max_depth if max_depth > 0 else 0,
            }
        
        return stats
    
    def is_healthy(self) -> bool:
        """Check if queue is healthy.
        
        Returns:
            True if Redis is available and queues are operational
        """
        return self.connection_manager.is_healthy()
    
    def close(self) -> None:
        """Close queue connections."""
        self.connection_manager.close()
    
    def __enter__(self) -> AudioRAGQueue:
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
