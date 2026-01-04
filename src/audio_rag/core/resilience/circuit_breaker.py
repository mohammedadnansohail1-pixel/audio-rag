"""Circuit breaker pattern implementation for external services."""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TypeVar, ParamSpec
from functools import wraps

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    name: str
    failure_threshold: int = 5
    success_threshold: int = 2
    reset_timeout: float = 30.0
    expected_exceptions: tuple = field(default_factory=lambda: (Exception,))


@dataclass
class CircuitStats:
    """Circuit breaker statistics."""
    failures: int = 0
    successes: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    total_calls: int = 0
    total_failures: int = 0


class CircuitBreakerError(Exception):
    """Raised when circuit is open."""
    
    def __init__(self, breaker_name: str, time_remaining: float):
        self.breaker_name = breaker_name
        self.time_remaining = time_remaining
        super().__init__(
            f"Circuit breaker '{breaker_name}' is OPEN. "
            f"Retry in {time_remaining:.1f}s"
        )


class CircuitBreaker:
    """
    Circuit breaker for protecting external service calls.
    
    States:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Service failing, calls rejected immediately
    - HALF_OPEN: Testing if service recovered
    
    Transitions:
    - CLOSED → OPEN: After failure_threshold consecutive failures
    - OPEN → HALF_OPEN: After reset_timeout seconds
    - HALF_OPEN → CLOSED: After success_threshold consecutive successes
    - HALF_OPEN → OPEN: On any failure
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._opened_at: Optional[float] = None
    
    @property
    def state(self) -> CircuitState:
        """Get current state, checking for timeout transition."""
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to(CircuitState.HALF_OPEN)
        return self._state
    
    @property
    def stats(self) -> CircuitStats:
        """Get current statistics."""
        return self._stats
    
    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time passed to try recovery."""
        if self._opened_at is None:
            return False
        return time.time() - self._opened_at >= self.config.reset_timeout
    
    def _time_remaining(self) -> float:
        """Time until circuit attempts reset."""
        if self._opened_at is None:
            return 0.0
        elapsed = time.time() - self._opened_at
        return max(0.0, self.config.reset_timeout - elapsed)
    
    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state with logging."""
        old_state = self._state
        self._state = new_state
        
        if new_state == CircuitState.OPEN:
            self._opened_at = time.time()
        elif new_state == CircuitState.CLOSED:
            self._stats.consecutive_failures = 0
            self._stats.consecutive_successes = 0
        
        logger.warning(
            f"Circuit breaker '{self.config.name}' "
            f"transitioned: {old_state.value} → {new_state.value}"
        )
    
    def record_success(self) -> None:
        """Record successful call."""
        current_state = self.state  # Trigger lazy state transition
        self._stats.successes += 1
        self._stats.consecutive_successes += 1
        self._stats.consecutive_failures = 0
        self._stats.last_success_time = time.time()
        self._stats.total_calls += 1
        
        if current_state == CircuitState.HALF_OPEN:
            if self._stats.consecutive_successes >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED)
    
    def record_failure(self, exception: Exception) -> None:
        """Record failed call."""
        self._stats.failures += 1
        self._stats.consecutive_failures += 1
        self._stats.consecutive_successes = 0
        self._stats.last_failure_time = time.time()
        self._stats.total_calls += 1
        self._stats.total_failures += 1
        
        logger.warning(
            f"Circuit breaker '{self.config.name}' "
            f"failure #{self._stats.consecutive_failures}: {exception}"
        )
        
        if self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open immediately opens
            self._transition_to(CircuitState.OPEN)
        elif self._state == CircuitState.CLOSED:
            if self._stats.consecutive_failures >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)
    
    def call(self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        """Execute function with circuit breaker protection."""
        state = self.state  # This may transition OPEN → HALF_OPEN
        
        if state == CircuitState.OPEN:
            raise CircuitBreakerError(
                self.config.name,
                self._time_remaining()
            )
        
        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except self.config.expected_exceptions as e:
            self.record_failure(e)
            raise
    
    async def call_async(
        self, 
        func: Callable[P, T], 
        *args: P.args, 
        **kwargs: P.kwargs
    ) -> T:
        """Execute async function with circuit breaker protection."""
        state = self.state
        
        if state == CircuitState.OPEN:
            raise CircuitBreakerError(
                self.config.name,
                self._time_remaining()
            )
        
        try:
            result = await func(*args, **kwargs)
            self.record_success()
            return result
        except self.config.expected_exceptions as e:
            self.record_failure(e)
            raise
    
    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        self._transition_to(CircuitState.CLOSED)
        self._stats = CircuitStats()
        self._opened_at = None
        logger.info(f"Circuit breaker '{self.config.name}' manually reset")


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    success_threshold: int = 2,
    reset_timeout: float = 30.0,
    expected_exceptions: tuple = (Exception,),
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to apply circuit breaker to a function.
    
    Usage:
        @circuit_breaker("redis", failure_threshold=3)
        def call_redis():
            ...
    """
    config = CircuitBreakerConfig(
        name=name,
        failure_threshold=failure_threshold,
        success_threshold=success_threshold,
        reset_timeout=reset_timeout,
        expected_exceptions=expected_exceptions,
    )
    breaker = CircuitBreaker(config)
    
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return breaker.call(func, *args, **kwargs)
        
        # Attach breaker for inspection/testing
        wrapper.circuit_breaker = breaker  # type: ignore
        return wrapper
    
    return decorator


# Pre-configured breakers for common services
REDIS_BREAKER_CONFIG = CircuitBreakerConfig(
    name="redis",
    failure_threshold=5,
    success_threshold=2,
    reset_timeout=30.0,
    expected_exceptions=(ConnectionError, TimeoutError, OSError),
)

QDRANT_BREAKER_CONFIG = CircuitBreakerConfig(
    name="qdrant",
    failure_threshold=3,
    success_threshold=2,
    reset_timeout=60.0,
    expected_exceptions=(ConnectionError, TimeoutError, Exception),
)

HUGGINGFACE_BREAKER_CONFIG = CircuitBreakerConfig(
    name="huggingface",
    failure_threshold=3,
    success_threshold=1,
    reset_timeout=120.0,
    expected_exceptions=(ConnectionError, TimeoutError, OSError),
)
