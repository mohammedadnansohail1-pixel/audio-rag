"""Redis connection manager with retry and circuit breaker.

Handles:
- Connection pooling
- Automatic reconnection with exponential backoff
- Circuit breaker for failing connections
- Health checks
"""

from __future__ import annotations

import logging
import threading
import time
from enum import StrEnum
from typing import TYPE_CHECKING

from redis import Redis
from redis.exceptions import ConnectionError as RedisConnError
from redis.exceptions import RedisError, TimeoutError as RedisTimeoutError

from audio_rag.queue.config import QueueConfig, RedisConfig
from audio_rag.queue.exceptions import RedisConnectionError, RedisOperationError

if TYPE_CHECKING:
    from redis.client import Pipeline

logger = logging.getLogger(__name__)


class CircuitState(StrEnum):
    """Circuit breaker states."""
    
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Failing, requests rejected immediately
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker for Redis connections.
    
    Prevents cascading failures by failing fast when Redis is unavailable.
    
    States:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Service is down, reject requests immediately
    - HALF_OPEN: Testing recovery, allow limited requests
    
    Transitions:
    - CLOSED -> OPEN: After `failure_threshold` consecutive failures
    - OPEN -> HALF_OPEN: After `recovery_timeout` seconds
    - HALF_OPEN -> CLOSED: After `success_threshold` consecutive successes
    - HALF_OPEN -> OPEN: After any failure
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 2,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._lock = threading.Lock()
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN:
                if (
                    self._last_failure_time is not None
                    and time.time() - self._last_failure_time >= self.recovery_timeout
                ):
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    logger.info("Circuit breaker: OPEN -> HALF_OPEN")
            
            return self._state
    
    @property
    def is_available(self) -> bool:
        """Check if requests should be allowed."""
        return self.state != CircuitState.OPEN
    
    def record_success(self) -> None:
        """Record a successful operation."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info("Circuit breaker: HALF_OPEN -> CLOSED")
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0
    
    def record_failure(self) -> None:
        """Record a failed operation."""
        with self._lock:
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._state = CircuitState.OPEN
                logger.warning("Circuit breaker: HALF_OPEN -> OPEN")
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(
                        f"Circuit breaker: CLOSED -> OPEN "
                        f"(failures={self._failure_count})"
                    )
    
    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None


class RedisConnectionManager:
    """Manages Redis connections with retry and circuit breaker.
    
    Features:
    - Lazy connection initialization
    - Automatic reconnection with exponential backoff
    - Circuit breaker to prevent cascading failures
    - Thread-safe
    - Health check support
    
    Example:
        manager = RedisConnectionManager(config.redis)
        
        # Get connection (auto-connects if needed)
        conn = manager.get_connection()
        
        # Use connection
        conn.set("key", "value")
        
        # Health check
        if manager.is_healthy():
            print("Redis is available")
    """
    
    def __init__(
        self,
        config: RedisConfig | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        """Initialize connection manager.
        
        Args:
            config: Redis configuration. Uses defaults if not provided.
            circuit_breaker: Custom circuit breaker. Creates default if not provided.
        """
        self.config = config or RedisConfig()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        
        self._connection: Redis | None = None
        self._lock = threading.Lock()
    
    @classmethod
    def from_queue_config(cls, config: QueueConfig) -> RedisConnectionManager:
        """Create manager from queue configuration."""
        return cls(config=config.redis)
    
    @classmethod
    def from_url(cls, url: str) -> RedisConnectionManager:
        """Create manager from Redis URL.
        
        Args:
            url: Redis URL (e.g., redis://localhost:6379/0)
        """
        # Parse URL to extract components
        # Simple parsing - for complex URLs use redis.connection.parse_url
        config = RedisConfig()
        
        if url.startswith("redis://"):
            url = url[8:]
        
        # Handle password
        if "@" in url:
            auth, url = url.split("@", 1)
            if ":" in auth:
                config.password = auth.split(":", 1)[1]
        
        # Handle host:port/db
        if "/" in url:
            host_port, db = url.rsplit("/", 1)
            config.db = int(db)
        else:
            host_port = url
        
        if ":" in host_port:
            config.host, port_str = host_port.split(":", 1)
            config.port = int(port_str)
        else:
            config.host = host_port
        
        return cls(config=config)
    
    def get_connection(self) -> Redis:
        """Get Redis connection, creating if needed.
        
        Returns:
            Active Redis connection
            
        Raises:
            RedisConnectionError: If connection fails after retries
        """
        # Check circuit breaker
        if not self.circuit_breaker.is_available:
            raise RedisConnectionError(
                "Circuit breaker is open, Redis unavailable",
                host=self.config.host,
                port=self.config.port,
            )
        
        with self._lock:
            if self._connection is not None:
                # Verify existing connection is alive
                try:
                    self._connection.ping()
                    self.circuit_breaker.record_success()
                    return self._connection
                except RedisError:
                    # Connection dead, will reconnect
                    self._connection = None
            
            # Create new connection with retry
            return self._connect_with_retry()
    
    def _connect_with_retry(self) -> Redis:
        """Connect to Redis with exponential backoff retry.
        
        Returns:
            Active Redis connection
            
        Raises:
            RedisConnectionError: If all retries fail
        """
        last_error: Exception | None = None
        
        for attempt in range(self.config.max_retries):
            try:
                conn = Redis(
                    host=self.config.host,
                    port=self.config.port,
                    db=self.config.db,
                    password=self.config.password,
                    socket_timeout=self.config.socket_timeout,
                    socket_connect_timeout=self.config.socket_connect_timeout,
                    decode_responses=True,  # Return strings, not bytes
                )
                
                # Verify connection
                conn.ping()
                
                self._connection = conn
                self.circuit_breaker.record_success()
                
                logger.info(
                    f"Connected to Redis at {self.config.host}:{self.config.port}"
                )
                return conn
                
            except (RedisConnError, RedisTimeoutError, OSError) as e:
                last_error = e
                self.circuit_breaker.record_failure()
                
                if attempt < self.config.max_retries - 1:
                    # Exponential backoff with jitter
                    backoff = self.config.retry_backoff * (2 ** attempt)
                    # Add jitter (±25%)
                    jitter = backoff * 0.25 * (2 * (hash(str(time.time())) % 100) / 100 - 1)
                    sleep_time = backoff + jitter
                    
                    logger.warning(
                        f"Redis connection failed (attempt {attempt + 1}/{self.config.max_retries}), "
                        f"retrying in {sleep_time:.1f}s: {e}"
                    )
                    time.sleep(sleep_time)
        
        # All retries failed
        raise RedisConnectionError(
            f"Failed to connect to Redis after {self.config.max_retries} attempts: {last_error}",
            host=self.config.host,
            port=self.config.port,
            attempts=self.config.max_retries,
        )
    
    def execute(self, operation: str, *args, **kwargs):
        """Execute a Redis operation with error handling.
        
        Args:
            operation: Redis command name (e.g., 'get', 'set', 'hset')
            *args: Command arguments
            **kwargs: Command keyword arguments
            
        Returns:
            Command result
            
        Raises:
            RedisConnectionError: If connection fails
            RedisOperationError: If operation fails
        """
        conn = self.get_connection()
        
        try:
            cmd = getattr(conn, operation)
            result = cmd(*args, **kwargs)
            self.circuit_breaker.record_success()
            return result
        except (RedisConnError, RedisTimeoutError) as e:
            self.circuit_breaker.record_failure()
            self._connection = None  # Force reconnect
            raise RedisConnectionError(
                f"Redis connection lost during operation: {e}",
                host=self.config.host,
                port=self.config.port,
            ) from e
        except RedisError as e:
            raise RedisOperationError(
                f"Redis operation failed: {e}",
                operation=operation,
                key=str(args[0]) if args else None,
            ) from e
    
    def pipeline(self) -> Pipeline:
        """Get a Redis pipeline for batched operations.
        
        Returns:
            Redis pipeline object
            
        Raises:
            RedisConnectionError: If connection fails
        """
        conn = self.get_connection()
        return conn.pipeline()
    
    def is_healthy(self) -> bool:
        """Check if Redis connection is healthy.
        
        Returns:
            True if Redis is reachable and responding
        """
        try:
            conn = self.get_connection()
            return conn.ping()
        except (RedisConnectionError, RedisError):
            return False
    
    def close(self) -> None:
        """Close the Redis connection."""
        with self._lock:
            if self._connection is not None:
                try:
                    self._connection.close()
                except RedisError:
                    pass  # Ignore errors on close
                self._connection = None
                logger.info("Redis connection closed")
    
    def __enter__(self) -> RedisConnectionManager:
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - close connection."""
        self.close()
