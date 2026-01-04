"""Resilience patterns for fault-tolerant operations."""

from audio_rag.core.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
    CircuitStats,
    circuit_breaker,
    REDIS_BREAKER_CONFIG,
    QDRANT_BREAKER_CONFIG,
    HUGGINGFACE_BREAKER_CONFIG,
)
from audio_rag.core.resilience.retry import (
    retry_with_backoff,
    retry_redis,
    retry_qdrant,
    retry_model_load,
    retry_network,
    retry_redis_async,
    retry_qdrant_async,
    RetryError,
)
from audio_rag.core.resilience.fallback import (
    FallbackChain,
    FallbackOption,
    FallbackExhaustedError,
    has_cuda,
    has_gpu_memory,
    create_asr_fallback_chain,
    create_embedding_fallback_chain,
)
from audio_rag.core.resilience.timeout import (
    TimeoutError,
    TimeoutConfig,
    DEFAULT_TIMEOUTS,
    async_timeout,
    with_timeout,
    sync_timeout,
    calculate_asr_timeout,
    calculate_diarization_timeout,
)

__all__ = [
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerError",
    "CircuitState",
    "CircuitStats",
    "circuit_breaker",
    "REDIS_BREAKER_CONFIG",
    "QDRANT_BREAKER_CONFIG",
    "HUGGINGFACE_BREAKER_CONFIG",
    # Retry
    "retry_with_backoff",
    "retry_redis",
    "retry_qdrant",
    "retry_model_load",
    "retry_network",
    "retry_redis_async",
    "retry_qdrant_async",
    "RetryError",
    # Fallback
    "FallbackChain",
    "FallbackOption",
    "FallbackExhaustedError",
    "has_cuda",
    "has_gpu_memory",
    "create_asr_fallback_chain",
    "create_embedding_fallback_chain",
    # Timeout
    "TimeoutError",
    "TimeoutConfig",
    "DEFAULT_TIMEOUTS",
    "async_timeout",
    "with_timeout",
    "sync_timeout",
    "calculate_asr_timeout",
    "calculate_diarization_timeout",
]
