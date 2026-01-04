"""GPU Worker for processing Audio RAG jobs.

Features:
- Model preloading (load once, process many)
- Memory budget enforcement
- Graceful shutdown
- Checkpoint-based recovery
- Health monitoring
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from redis import Redis
from rq import Worker
from rq.job import Job as RQJob

from audio_rag.queue.config import QueueConfig, WorkerConfig
from audio_rag.queue.connection import RedisConnectionManager
from audio_rag.queue.exceptions import (
    WorkerResourceError,
    WorkerShutdownError,
    WorkerStartupError,
)
from audio_rag.queue.job import (
    IngestJob,
    JobCheckpoint,
    JobResult,
    JobStage,
    JobStatus,
)

if TYPE_CHECKING:
    from audio_rag.pipeline import AudioRAG

logger = logging.getLogger(__name__)


# Redis keys
KEY_PREFIX = "audio_rag:"
JOB_STATUS_PREFIX = f"{KEY_PREFIX}job_status:"
CHECKPOINT_PREFIX = f"{KEY_PREFIX}checkpoint:"
WORKER_PREFIX = f"{KEY_PREFIX}worker:"


def get_gpu_memory_info() -> dict[str, float]:
    """Get GPU memory information.
    
    Returns:
        Dict with total_mb, used_mb, free_mb, utilization
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return {
                "total_mb": 0,
                "used_mb": 0,
                "free_mb": 0,
                "utilization": 0,
                "available": False,
            }
        
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        
        total_mb = total / (1024 * 1024)
        used_mb = reserved / (1024 * 1024)
        free_mb = (total - reserved) / (1024 * 1024)
        
        return {
            "total_mb": total_mb,
            "used_mb": used_mb,
            "free_mb": free_mb,
            "utilization": used_mb / total_mb if total_mb > 0 else 0,
            "available": True,
        }
    except ImportError:
        return {
            "total_mb": 0,
            "used_mb": 0,
            "free_mb": 0,
            "utilization": 0,
            "available": False,
        }


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    try:
        import torch
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("GPU memory cache cleared")
    except ImportError:
        pass


class GPUWorker:
    """GPU-enabled worker for Audio RAG jobs.
    
    Handles:
    - Model preloading at startup
    - Memory budget enforcement
    - Job processing with checkpoints
    - Graceful shutdown
    - Health reporting
    
    Example:
        worker = GPUWorker.from_config(config)
        worker.start()  # Blocks until shutdown
    """
    
    def __init__(
        self,
        connection_manager: RedisConnectionManager,
        config: QueueConfig,
        worker_id: str | None = None,
    ) -> None:
        """Initialize worker.
        
        Args:
            connection_manager: Redis connection manager
            config: Queue configuration
            worker_id: Unique worker ID (auto-generated if not provided)
        """
        self.connection_manager = connection_manager
        self.config = config
        self.worker_config = config.worker
        self.worker_id = worker_id or f"worker-{os.getpid()}"
        
        self._pipeline: AudioRAG | None = None
        self._rq_worker: Worker | None = None
        self._shutdown_event = threading.Event()
        self._running = False
        self._consecutive_failures = 0
    
    @classmethod
    def from_config(
        cls,
        config: QueueConfig | None = None,
        worker_id: str | None = None,
    ) -> GPUWorker:
        """Create worker from configuration."""
        config = config or QueueConfig()
        connection_manager = RedisConnectionManager(config.redis)
        return cls(connection_manager, config, worker_id)
    
    def _get_redis(self) -> Redis:
        """Get Redis connection."""
        return self.connection_manager.get_connection()
    
    def _check_resources(self) -> None:
        """Check if worker has sufficient resources.
        
        Raises:
            WorkerResourceError: If resources insufficient
        """
        gpu_info = get_gpu_memory_info()
        
        if gpu_info["available"]:
            if gpu_info["free_mb"] < 1000:  # Need at least 1GB free
                raise WorkerResourceError(
                    "Insufficient GPU memory",
                    worker_id=self.worker_id,
                    resource="gpu_memory",
                    required=1000,
                    available=gpu_info["free_mb"],
                    unit="MB",
                )
        
        # Check disk space
        try:
            import shutil
            disk = shutil.disk_usage("/")
            free_gb = disk.free / (1024 ** 3)
            if free_gb < 1:
                raise WorkerResourceError(
                    "Insufficient disk space",
                    worker_id=self.worker_id,
                    resource="disk",
                    required=1,
                    available=free_gb,
                    unit="GB",
                )
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
    
    def _load_pipeline(self) -> None:
        """Load the Audio RAG pipeline.
        
        Raises:
            WorkerStartupError: If pipeline loading fails
        """
        try:
            from audio_rag.pipeline import AudioRAG
            from audio_rag.config import load_config
            
            logger.info("Loading Audio RAG pipeline...")
            
            # Load config
            config = load_config()
            
            # Initialize pipeline (loads models)
            self._pipeline = AudioRAG(config)
            
            logger.info("Pipeline loaded successfully")
            
        except ImportError as e:
            raise WorkerStartupError(
                f"Failed to import pipeline: {e}",
                worker_id=self.worker_id,
                stage="import",
            ) from e
        except Exception as e:
            raise WorkerStartupError(
                f"Failed to load pipeline: {e}",
                worker_id=self.worker_id,
                stage="load_models",
            ) from e
    
    def _register_worker(self) -> None:
        """Register worker in Redis for monitoring."""
        key = f"{WORKER_PREFIX}{self.worker_id}"
        
        gpu_info = get_gpu_memory_info()
        
        data = {
            "worker_id": self.worker_id,
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "gpu_available": gpu_info["available"],
            "gpu_total_mb": gpu_info["total_mb"],
        }
        
        self._get_redis().setex(
            key,
            self.worker_config.heartbeat_interval * 3,  # TTL
            json.dumps(data),
        )
    
    def _update_heartbeat(self) -> None:
        """Update worker heartbeat in Redis."""
        key = f"{WORKER_PREFIX}{self.worker_id}"
        
        gpu_info = get_gpu_memory_info()
        
        data = {
            "worker_id": self.worker_id,
            "pid": os.getpid(),
            "last_heartbeat": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "gpu_used_mb": gpu_info["used_mb"],
            "gpu_free_mb": gpu_info["free_mb"],
            "consecutive_failures": self._consecutive_failures,
        }
        
        self._get_redis().setex(
            key,
            self.worker_config.heartbeat_interval * 3,
            json.dumps(data),
        )
    
    def _unregister_worker(self) -> None:
        """Unregister worker from Redis."""
        key = f"{WORKER_PREFIX}{self.worker_id}"
        self._get_redis().delete(key)
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def handle_signal(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self._shutdown_event.set()
        
        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)
    
    def _update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        stage: JobStage,
        error: str | None = None,
    ) -> None:
        """Update job status in Redis."""
        key = f"{JOB_STATUS_PREFIX}{job_id}"
        
        data = {
            "status": status.value,
            "stage": stage.value,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "error": error,
            "worker_id": self.worker_id,
        }
        
        self._get_redis().setex(
            key,
            self.config.result_ttl,
            json.dumps(data),
        )
    
    def _save_checkpoint(
        self,
        job_id: str,
        stage: JobStage,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Save job checkpoint for recovery."""
        checkpoint = JobCheckpoint(
            job_id=job_id,
            stage=stage,
            data=data or {},
        )
        
        key = f"{CHECKPOINT_PREFIX}{job_id}"
        self._get_redis().setex(
            key,
            self.config.checkpoint_ttl,
            checkpoint.to_json(),
        )
    
    def _get_checkpoint(self, job_id: str) -> JobCheckpoint | None:
        """Get job checkpoint if exists."""
        key = f"{CHECKPOINT_PREFIX}{job_id}"
        data = self._get_redis().get(key)
        
        if data:
            return JobCheckpoint.from_json(data)
        return None
    
    def start(self) -> None:
        """Start the worker (blocking).
        
        This method blocks until shutdown is requested.
        
        Raises:
            WorkerStartupError: If startup fails
        """
        logger.info(f"Starting worker {self.worker_id}...")
        
        try:
            # Check resources
            self._check_resources()
            
            # Load pipeline
            self._load_pipeline()
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            # Register worker
            self._register_worker()
            
            # Create RQ worker
            conn = self._get_redis()
            queues = [q.name for q in self.config.queues]
            
            self._rq_worker = Worker(
                queues=queues,
                connection=conn,
                name=self.worker_id,
            )
            
            self._running = True
            logger.info(f"Worker {self.worker_id} started, listening on queues: {queues}")
            
            # Start heartbeat thread
            heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                daemon=True,
            )
            heartbeat_thread.start()
            
            # Run worker (blocking)
            self._rq_worker.work(
                burst=False,
                logging_level=logging.INFO,
            )
            
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Worker error: {e}")
            raise
        finally:
            self.shutdown()
    
    def _heartbeat_loop(self) -> None:
        """Background thread for heartbeat updates."""
        while self._running and not self._shutdown_event.is_set():
            try:
                self._update_heartbeat()
            except Exception as e:
                logger.warning(f"Heartbeat update failed: {e}")
            
            self._shutdown_event.wait(self.worker_config.heartbeat_interval)
    
    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown the worker gracefully.
        
        Args:
            timeout: Maximum time to wait for shutdown
        """
        timeout = timeout or self.worker_config.shutdown_timeout
        logger.info(f"Shutting down worker {self.worker_id}...")
        
        self._running = False
        self._shutdown_event.set()
        
        # Stop RQ worker
        if self._rq_worker:
            try:
                self._rq_worker.request_stop(signal.SIGTERM, None)
            except Exception as e:
                logger.warning(f"Error stopping RQ worker: {e}")
        
        # Unload pipeline
        if self._pipeline:
            try:
                # Pipeline cleanup if available
                if hasattr(self._pipeline, "cleanup"):
                    self._pipeline.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up pipeline: {e}")
            self._pipeline = None
        
        # Clear GPU memory
        clear_gpu_memory()
        
        # Unregister worker
        try:
            self._unregister_worker()
        except Exception as e:
            logger.warning(f"Error unregistering worker: {e}")
        
        # Close Redis connection
        self.connection_manager.close()
        
        logger.info(f"Worker {self.worker_id} shutdown complete")


def process_ingest_job(job_data: dict[str, Any]) -> dict[str, Any]:
    """Process an ingestion job.
    
    This is the function called by RQ workers.
    
    Args:
        job_data: Serialized IngestJob data
        
    Returns:
        Serialized JobResult data
    """
    job = IngestJob.from_dict(job_data)
    started_at = datetime.now(timezone.utc)
    
    logger.info(f"Processing job {job.job_id} (tenant={job.tenant_id})")
    
    # Get Redis connection from current job
    rq_job = RQJob.fetch(job.job_id, connection=Redis())
    redis = rq_job.connection
    
    # Update status to running
    status_key = f"{JOB_STATUS_PREFIX}{job.job_id}"
    redis.setex(
        status_key,
        86400,
        json.dumps({
            "status": JobStatus.RUNNING.value,
            "stage": JobStage.VALIDATED.value,
            "updated_at": started_at.isoformat(),
        }),
    )
    
    try:
        # Import pipeline here (worker process has it loaded)
        from audio_rag.pipeline import AudioRAG
        from audio_rag.config import load_config
        
        # Load config and create pipeline
        config = load_config()
        
        # Apply job-specific overrides
        if job.config_overrides:
            # Merge overrides into config
            for key, value in job.config_overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Initialize pipeline
        pipeline = AudioRAG(config)
        
        # Process the audio file
        result = pipeline.ingest(
            audio_path=job.audio_path,
            collection_name=job.tenant_id,
            metadata=job.metadata,
        )
        
        completed_at = datetime.now(timezone.utc)
        
        # Build result
        job_result = JobResult(
            job_id=job.job_id,
            status=JobStatus.COMPLETED,
            stage=JobStage.COMPLETE,
            started_at=started_at,
            completed_at=completed_at,
            chunks_created=result.chunks_created if hasattr(result, "chunks_created") else 0,
            metrics={
                "audio_duration_seconds": getattr(result, "audio_duration", 0),
                "processing_time_seconds": (completed_at - started_at).total_seconds(),
            },
        )
        
        # Update status
        redis.setex(
            status_key,
            86400,
            json.dumps({
                "status": JobStatus.COMPLETED.value,
                "stage": JobStage.COMPLETE.value,
                "updated_at": completed_at.isoformat(),
            }),
        )
        
        logger.info(
            f"Job {job.job_id} completed: {job_result.chunks_created} chunks, "
            f"{job_result.duration_seconds:.1f}s"
        )
        
        return job_result.to_dict()
        
    except Exception as e:
        completed_at = datetime.now(timezone.utc)
        
        logger.error(f"Job {job.job_id} failed: {e}")
        
        # Build error result
        job_result = JobResult(
            job_id=job.job_id,
            status=JobStatus.FAILED,
            stage=JobStage.QUEUED,  # Or last checkpoint stage
            started_at=started_at,
            completed_at=completed_at,
            error_message=str(e),
            error_type=type(e).__name__,
        )
        
        # Update status
        redis.setex(
            status_key,
            86400,
            json.dumps({
                "status": JobStatus.FAILED.value,
                "stage": JobStage.QUEUED.value,
                "updated_at": completed_at.isoformat(),
                "error": str(e),
            }),
        )
        
        raise  # Re-raise so RQ marks job as failed


# CLI entry point
def main() -> None:
    """Main entry point for worker CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Audio RAG GPU Worker")
    parser.add_argument(
        "--worker-id",
        help="Unique worker ID",
    )
    parser.add_argument(
        "--config",
        help="Path to queue config file",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    # Load config
    config = QueueConfig()
    if args.config:
        # Load from file if provided
        pass  # TODO: implement config file loading
    
    # Create and start worker
    worker = GPUWorker.from_config(config, worker_id=args.worker_id)
    worker.start()


if __name__ == "__main__":
    main()
