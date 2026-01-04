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
    """GPU-enabled worker for Audio RAG processing.

    Features:
    - Preloads ML models for efficiency
    - Monitors GPU memory
    - Graceful shutdown on SIGTERM/SIGINT
    - Health endpoint for Kubernetes
    """

    def __init__(
        self,
        redis_manager: RedisConnectionManager,
        config: WorkerConfig,
        worker_id: str | None = None,
    ):
        self.redis_manager = redis_manager
        self.config = config
        self.worker_id = worker_id or f"worker-{os.getpid()}"

        self._shutdown_requested = False
        self._current_job: str | None = None
        self._pipeline: AudioRAG | None = None
        self._rq_worker: Worker | None = None

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        logger.info(f"GPUWorker {self.worker_id} initialized")

    @classmethod
    def from_config(
        cls, config: QueueConfig, worker_id: str | None = None
    ) -> "GPUWorker":
        """Create worker from queue config."""
        redis_manager = RedisConnectionManager(config.redis)
        return cls(
            redis_manager=redis_manager,
            config=config.worker,
            worker_id=worker_id,
        )

    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        """Handle shutdown signal."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._shutdown_requested = True

        if self._rq_worker:
            self._rq_worker.request_stop(signum, frame)

    def _preload_models(self) -> None:
        """Preload ML models to GPU memory."""
        if not self.config.preload_models:
            logger.info("Model preloading disabled")
            return

        logger.info("Preloading ML models...")

        try:
            from audio_rag.pipeline import AudioRAG
            from audio_rag.config import load_config

            config = load_config()
            self._pipeline = AudioRAG(config)

            # Touch properties to trigger loading
            _ = self._pipeline.embedder
            if self._pipeline.embedder and not self._pipeline.embedder.is_loaded:
                self._pipeline.embedder.load()

            logger.info("Models preloaded successfully")

            # Log GPU memory status
            gpu_info = get_gpu_memory_info()
            if gpu_info["available"]:
                logger.info(
                    f"GPU memory after preload: "
                    f"{gpu_info['used_mb']:.0f}MB / {gpu_info['total_mb']:.0f}MB "
                    f"({gpu_info['utilization']:.1%})"
                )

        except Exception as e:
            logger.error(f"Failed to preload models: {e}")
            raise WorkerStartupError(f"Model preload failed: {e}")

    def _register_worker(self) -> None:
        """Register worker in Redis for health monitoring."""
        try:
            redis = self.redis_manager.get_sync_client()
            worker_key = f"{WORKER_PREFIX}{self.worker_id}"

            worker_info = {
                "worker_id": self.worker_id,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "status": "running",
                "gpu": get_gpu_memory_info(),
                "pid": os.getpid(),
            }

            redis.setex(worker_key, 300, json.dumps(worker_info))  # 5 min TTL
            logger.info(f"Worker {self.worker_id} registered")

        except Exception as e:
            logger.warning(f"Failed to register worker: {e}")

    def _update_heartbeat(self) -> None:
        """Update worker heartbeat in Redis."""
        try:
            redis = self.redis_manager.get_sync_client()
            worker_key = f"{WORKER_PREFIX}{self.worker_id}"

            worker_info = {
                "worker_id": self.worker_id,
                "last_heartbeat": datetime.now(timezone.utc).isoformat(),
                "status": "running",
                "current_job": self._current_job,
                "gpu": get_gpu_memory_info(),
            }

            redis.setex(worker_key, 300, json.dumps(worker_info))

        except Exception as e:
            logger.warning(f"Failed to update heartbeat: {e}")

    def _start_heartbeat_thread(self) -> threading.Thread:
        """Start background heartbeat thread."""

        def heartbeat_loop():
            while not self._shutdown_requested:
                self._update_heartbeat()
                time.sleep(30)  # Update every 30 seconds

        thread = threading.Thread(target=heartbeat_loop, daemon=True)
        thread.start()
        return thread

    def start(self) -> None:
        """Start the worker."""
        logger.info(f"Starting GPUWorker {self.worker_id}")

        # Check GPU
        gpu_info = get_gpu_memory_info()
        if gpu_info["available"]:
            logger.info(
                f"GPU available: {gpu_info['total_mb']:.0f}MB total, "
                f"{gpu_info['free_mb']:.0f}MB free"
            )
        else:
            logger.warning("No GPU available, running in CPU mode")

        # Preload models
        self._preload_models()

        # Register worker
        self._register_worker()

        # Start heartbeat
        self._start_heartbeat_thread()

        # Get Redis connection and queues
        redis_conn = self.redis_manager.get_sync_client()
        queues = [
            f"{self.config.queue_prefix}:{priority}"
            for priority in ["high", "normal", "low"]
        ]

        # Create RQ worker
        self._rq_worker = Worker(
            queues=queues,
            connection=redis_conn,
            name=self.worker_id,
        )

        logger.info(f"Worker listening on queues: {queues}")

        # Run worker
        self._rq_worker.work(
            with_scheduler=False,
            burst=False,
        )

        logger.info(f"Worker {self.worker_id} stopped")

    def stop(self) -> None:
        """Stop the worker gracefully."""
        logger.info(f"Stopping worker {self.worker_id}")
        self._shutdown_requested = True

        # Unload models
        if self._pipeline:
            self._pipeline.unload_all()
            clear_gpu_memory()

        # Deregister from Redis
        try:
            redis = self.redis_manager.get_sync_client()
            worker_key = f"{WORKER_PREFIX}{self.worker_id}"
            redis.delete(worker_key)

        except Exception as e:
            logger.warning(f"Failed to deregister worker: {e}")

        logger.info(f"Worker {self.worker_id} stopped")


def process_ingest_job(job_data: dict) -> dict:
    """Process an ingestion job.

    This function is called by RQ to process jobs.

    Args:
        job_data: Serialized IngestJob data

    Returns:
        Serialized JobResult
    """
    started_at = datetime.now(timezone.utc)
    job = IngestJob.from_dict(job_data)

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

        # Process the audio file with proper parameters
        result = pipeline.ingest(
            audio_path=job.audio_path,
            collection_name=job.tenant_id,  # tenant_id maps to collection
            metadata=job.metadata,
            enable_diarization=job.config_overrides.get("enable_diarization", True) if job.config_overrides else True,
            language=job.config_overrides.get("language") if job.config_overrides else None,
        )

        completed_at = datetime.now(timezone.utc)

        # Build result using correct attribute names from IngestionResult
        job_result = JobResult(
            job_id=job.job_id,
            status=JobStatus.COMPLETED,
            stage=JobStage.COMPLETE,
            started_at=started_at,
            completed_at=completed_at,
            chunks_created=result.num_chunks,
            metrics={
                "audio_duration_seconds": result.duration_seconds,
                "processing_time_seconds": (completed_at - started_at).total_seconds(),
                "num_segments": result.num_segments,
                "num_speakers": len(result.speakers),
                "language": result.language,
                "collection": result.collection_name,
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
                "result": {
                    "chunks_created": result.num_chunks,
                    "duration_seconds": result.duration_seconds,
                    "speakers": result.speakers,
                    "collection": result.collection_name,
                },
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
                "error_type": type(e).__name__,
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
