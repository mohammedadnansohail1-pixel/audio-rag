"""Timeout patterns for async operations."""

import asyncio
import logging
import signal
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Optional, TypeVar, ParamSpec

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class TimeoutError(Exception):
    """Raised when operation exceeds timeout."""
    
    def __init__(self, operation: str, timeout: float):
        self.operation = operation
        self.timeout = timeout
        super().__init__(
            f"Operation '{operation}' timed out after {timeout:.1f}s"
        )


@dataclass
class TimeoutConfig:
    """Timeout configuration for different operations."""
    # Health checks
    health_check: float = 5.0
    
    # Redis operations
    redis_connect: float = 5.0
    redis_operation: float = 10.0
    
    # Qdrant operations
    qdrant_connect: float = 10.0
    qdrant_search: float = 30.0
    qdrant_upsert: float = 60.0
    
    # ML model operations
    model_load: float = 300.0  # 5 minutes
    asr_per_minute: float = 30.0  # 30s per minute of audio
    diarization_per_minute: float = 20.0
    embedding_batch: float = 60.0
    
    # API operations
    file_upload: float = 300.0
    job_status: float = 5.0
    query: float = 30.0


# Default timeout configuration
DEFAULT_TIMEOUTS = TimeoutConfig()


async def async_timeout(
    coro,
    timeout: float,
    operation: str = "operation",
) -> T:
    """
    Execute coroutine with timeout.
    
    Args:
        coro: Coroutine to execute
        timeout: Timeout in seconds
        operation: Operation name for error message
    
    Returns:
        Result of coroutine
    
    Raises:
        TimeoutError: If operation exceeds timeout
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(operation, timeout)


def with_timeout(
    timeout: float,
    operation: Optional[str] = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to add timeout to async function.
    
    Usage:
        @with_timeout(30.0, "search_vectors")
        async def search():
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        op_name = operation or func.__name__
        
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.error(f"Timeout: {op_name} exceeded {timeout}s")
                raise TimeoutError(op_name, timeout)
        
        return wrapper
    return decorator


@contextmanager
def sync_timeout(timeout: float, operation: str = "operation"):
    """
    Context manager for sync timeout using signals (Unix only).
    
    Usage:
        with sync_timeout(30.0, "model_inference"):
            model.transcribe(audio)
    
    Note: Only works on Unix systems and main thread.
    """
    def handler(signum, frame):
        raise TimeoutError(operation, timeout)
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    
    try:
        yield
    finally:
        # Restore the old handler
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def calculate_asr_timeout(
    audio_duration_seconds: float,
    config: TimeoutConfig = DEFAULT_TIMEOUTS,
) -> float:
    """
    Calculate appropriate timeout for ASR based on audio duration.
    
    Args:
        audio_duration_seconds: Duration of audio file
        config: Timeout configuration
    
    Returns:
        Timeout in seconds (minimum 60s, scales with duration)
    """
    audio_minutes = audio_duration_seconds / 60
    timeout = audio_minutes * config.asr_per_minute
    return max(60.0, min(timeout, 3600.0))  # 1 min to 1 hour


def calculate_diarization_timeout(
    audio_duration_seconds: float,
    config: TimeoutConfig = DEFAULT_TIMEOUTS,
) -> float:
    """
    Calculate appropriate timeout for diarization based on audio duration.
    
    Args:
        audio_duration_seconds: Duration of audio file
        config: Timeout configuration
    
    Returns:
        Timeout in seconds
    """
    audio_minutes = audio_duration_seconds / 60
    timeout = audio_minutes * config.diarization_per_minute
    return max(30.0, min(timeout, 1800.0))  # 30s to 30 min
