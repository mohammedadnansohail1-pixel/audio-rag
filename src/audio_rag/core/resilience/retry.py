"""Retry logic with exponential backoff using tenacity."""

import logging
from typing import Callable, TypeVar, ParamSpec, Type, Tuple, Union

from tenacity import (
    retry,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    wait_exponential_jitter,
    wait_fixed,
    retry_if_exception_type,
    before_log,
    after_log,
    RetryError,
)

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")

# Re-export for convenience
__all__ = [
    "retry_with_backoff",
    "retry_redis",
    "retry_qdrant",
    "retry_model_load",
    "retry_network",
    "RetryError",
]


def retry_with_backoff(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 30.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    log_retries: bool = True,
):
    """
    Generic retry decorator with exponential backoff and jitter.
    
    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        exceptions: Tuple of exception types to retry on
        log_retries: Whether to log retry attempts
    
    Usage:
        @retry_with_backoff(max_attempts=3, exceptions=(ConnectionError,))
        def call_service():
            ...
    """
    retry_kwargs = {
        "stop": stop_after_attempt(max_attempts),
        "wait": wait_exponential_jitter(initial=min_wait, max=max_wait),
        "retry": retry_if_exception_type(exceptions),
        "reraise": True,
    }
    
    if log_retries:
        retry_kwargs["before"] = before_log(logger, logging.WARNING)
        retry_kwargs["after"] = after_log(logger, logging.WARNING)
    
    return retry(**retry_kwargs)


def retry_redis(func: Callable[P, T]) -> Callable[P, T]:
    """
    Retry decorator for Redis operations.
    
    - 3 attempts
    - 0.5s - 5s exponential backoff with jitter
    - Retries on connection and timeout errors
    """
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=0.5, max=5),
        retry=retry_if_exception_type((
            ConnectionError,
            TimeoutError,
            OSError,
        )),
        before=before_log(logger, logging.DEBUG),
        reraise=True,
    )(func)


def retry_qdrant(func: Callable[P, T]) -> Callable[P, T]:
    """
    Retry decorator for Qdrant operations.
    
    - 3 attempts
    - 1s - 30s exponential backoff with jitter
    - Retries on connection errors
    """
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=30),
        retry=retry_if_exception_type((
            ConnectionError,
            TimeoutError,
            OSError,
        )),
        before=before_log(logger, logging.DEBUG),
        reraise=True,
    )(func)


def retry_model_load(func: Callable[P, T]) -> Callable[P, T]:
    """
    Retry decorator for model loading operations.
    
    - 2 attempts (models are expensive to retry)
    - 5s - 60s exponential backoff
    - Retries on download/IO errors
    """
    return retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=5, min=5, max=60),
        retry=retry_if_exception_type((
            ConnectionError,
            TimeoutError,
            OSError,
            IOError,
        )),
        before=before_log(logger, logging.WARNING),
        reraise=True,
    )(func)


def retry_network(
    max_attempts: int = 5,
    max_delay: float = 60.0,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Retry decorator for network operations with configurable limits.
    
    Uses exponential backoff with jitter to avoid thundering herd.
    
    Args:
        max_attempts: Maximum retry attempts
        max_delay: Maximum total delay (seconds)
    """
    return retry(
        stop=stop_after_attempt(max_attempts) | stop_after_delay(max_delay),
        wait=wait_exponential_jitter(initial=1, max=30),
        retry=retry_if_exception_type((
            ConnectionError,
            TimeoutError,
            OSError,
        )),
        before=before_log(logger, logging.DEBUG),
        reraise=True,
    )


# Async versions
def retry_redis_async(func: Callable[P, T]) -> Callable[P, T]:
    """Async version of retry_redis."""
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=0.5, max=5),
        retry=retry_if_exception_type((
            ConnectionError,
            TimeoutError,
            OSError,
        )),
        before=before_log(logger, logging.DEBUG),
        reraise=True,
    )(func)


def retry_qdrant_async(func: Callable[P, T]) -> Callable[P, T]:
    """Async version of retry_qdrant."""
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=30),
        retry=retry_if_exception_type((
            ConnectionError,
            TimeoutError,
            OSError,
        )),
        before=before_log(logger, logging.DEBUG),
        reraise=True,
    )(func)
