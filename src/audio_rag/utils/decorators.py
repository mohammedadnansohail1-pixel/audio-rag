"""Reusable decorators for logging, timing, retry, and validation."""

import functools
import time
import logging
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)


def timed(func: Callable[P, R]) -> Callable[P, R]:
    """Log execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__qualname__} completed in {elapsed:.2f}s")
        return result
    return wrapper


def logged(func: Callable[P, R]) -> Callable[P, R]:
    """Log function entry and exit."""
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        logger.debug(f"{func.__qualname__} called")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__qualname__} succeeded")
            return result
        except Exception as e:
            logger.error(f"{func.__qualname__} failed: {e}")
            raise
    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    """Retry a function on failure with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between retries (doubles each attempt)
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_exception = None
            current_delay = delay
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(
                            f"{func.__qualname__} attempt {attempt}/{max_attempts} failed: {e}. "
                            f"Retrying in {current_delay:.1f}s"
                        )
                        time.sleep(current_delay)
                        current_delay *= 2
                    else:
                        logger.error(f"{func.__qualname__} failed after {max_attempts} attempts")
            
            raise last_exception
        return wrapper
    return decorator


def require_loaded(func: Callable[P, R]) -> Callable[P, R]:
    """Ensure model is loaded before method execution.
    
    For use with classes that have `is_loaded` property and `load()` method.
    """
    @functools.wraps(func)
    def wrapper(self, *args: P.args, **kwargs: P.kwargs) -> R:
        if not self.is_loaded:
            logger.info(f"{self.__class__.__name__}: Auto-loading model")
            self.load()
        return func(self, *args, **kwargs)
    return wrapper
