"""FastAPI dependencies for auth, rate limiting, and context injection."""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status
from pydantic import BaseModel

from audio_rag.api.config import APIConfig, DEFAULT_API_CONFIG
from audio_rag.queue import AudioRAGQueue

logger = logging.getLogger(__name__)


# ============================================================================
# Request Context
# ============================================================================

class RequestContext(BaseModel):
    """Context for the current request."""
    
    request_id: str
    tenant_id: str
    tier: str
    api_key_name: str


def get_request_id(
    x_request_id: Annotated[str | None, Header()] = None,
) -> str:
    """Get or generate request ID for tracing."""
    if x_request_id:
        return x_request_id
    return str(uuid.uuid4())


# ============================================================================
# Authentication
# ============================================================================

class APIKeyData(BaseModel):
    """Data associated with an API key."""
    
    tenant_id: str
    tier: str
    name: str


async def get_api_key(
    request: Request,
    config: Annotated[APIConfig, Depends(lambda: DEFAULT_API_CONFIG)],
) -> APIKeyData:
    """Validate API key from header.
    
    Raises:
        HTTPException: 401 if missing, 403 if invalid
    """
    api_key = request.headers.get(config.auth.header_name)
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "API key required",
                "code": "AUTH_REQUIRED",
                "header": config.auth.header_name,
            },
        )
    
    # In production: look up in Redis/DB with hashed key
    # For now: use dev keys from config
    key_data = config.auth.dev_api_keys.get(api_key)
    
    if not key_data:
        logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "Invalid API key",
                "code": "AUTH_INVALID",
            },
        )
    
    return APIKeyData(**key_data)


async def get_request_context(
    request_id: Annotated[str, Depends(get_request_id)],
    api_key: Annotated[APIKeyData, Depends(get_api_key)],
) -> RequestContext:
    """Get full request context including auth info."""
    return RequestContext(
        request_id=request_id,
        tenant_id=api_key.tenant_id,
        tier=api_key.tier,
        api_key_name=api_key.name,
    )


# ============================================================================
# Rate Limiting
# ============================================================================

class RateLimitResult(BaseModel):
    """Result of rate limit check."""
    
    allowed: bool
    limit: int
    remaining: int
    reset_at: int  # Unix timestamp
    retry_after: int | None = None  # Seconds until reset


async def check_rate_limit(
    request: Request,
    api_key: Annotated[APIKeyData, Depends(get_api_key)],
    config: Annotated[APIConfig, Depends(lambda: DEFAULT_API_CONFIG)],
    endpoint_type: str = "query",
) -> RateLimitResult:
    """Check rate limit for current request.
    
    Uses Redis sliding window counter.
    """
    # Get rate limit for this tier and endpoint
    tier_limits = config.rate_limits.get_tier(api_key.tier)
    limit_config = getattr(tier_limits, endpoint_type, tier_limits.query)
    
    # Build rate limit key
    key = f"ratelimit:{api_key.tenant_id}:{endpoint_type}"
    
    # Get Redis from queue
    queue: AudioRAGQueue | None = getattr(request.app.state, "queue", None)
    
    if not queue or not queue.is_healthy():
        # Redis down - fail open or closed based on policy
        # Fail open for now (allow request)
        logger.warning("Rate limiting unavailable - Redis down")
        return RateLimitResult(
            allowed=True,
            limit=limit_config.requests,
            remaining=limit_config.requests,
            reset_at=int(time.time()) + limit_config.window_seconds,
        )
    
    redis = queue.connection_manager.get_connection()
    current_time = int(time.time())
    window_start = current_time - limit_config.window_seconds
    
    # Sliding window using sorted set
    pipe = redis.pipeline()
    pipe.zremrangebyscore(key, 0, window_start)  # Remove old entries
    pipe.zadd(key, {str(current_time): current_time})  # Add current
    pipe.zcard(key)  # Count entries
    pipe.expire(key, limit_config.window_seconds)  # Set TTL
    results = pipe.execute()
    
    count = results[2]
    remaining = max(0, limit_config.requests - count)
    reset_at = current_time + limit_config.window_seconds
    
    if count > limit_config.requests:
        return RateLimitResult(
            allowed=False,
            limit=limit_config.requests,
            remaining=0,
            reset_at=reset_at,
            retry_after=limit_config.window_seconds,
        )
    
    return RateLimitResult(
        allowed=True,
        limit=limit_config.requests,
        remaining=remaining,
        reset_at=reset_at,
    )


def rate_limit_dependency(endpoint_type: str = "query"):
    """Create a rate limit dependency for a specific endpoint type.
    
    Usage:
        @router.post("/query", dependencies=[Depends(rate_limit_dependency("query"))])
    """
    async def _check(
        request: Request,
        api_key: Annotated[APIKeyData, Depends(get_api_key)],
        config: Annotated[APIConfig, Depends(lambda: DEFAULT_API_CONFIG)],
    ) -> None:
        result = await check_rate_limit(request, api_key, config, endpoint_type)
        
        # Add rate limit headers to response
        request.state.rate_limit = result
        
        if not result.allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "code": "RATE_LIMITED",
                    "limit": result.limit,
                    "reset_at": result.reset_at,
                    "retry_after": result.retry_after,
                },
                headers={
                    "Retry-After": str(result.retry_after),
                    "X-RateLimit-Limit": str(result.limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(result.reset_at),
                },
            )
    
    return _check


# ============================================================================
# Queue Access
# ============================================================================

async def get_queue(request: Request) -> AudioRAGQueue:
    """Get the job queue from app state.
    
    Raises:
        HTTPException: 503 if queue unavailable
    """
    queue: AudioRAGQueue | None = getattr(request.app.state, "queue", None)
    
    if not queue:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Queue service unavailable",
                "code": "SERVICE_UNAVAILABLE",
            },
        )
    
    if not queue.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Queue service unhealthy",
                "code": "SERVICE_UNAVAILABLE",
            },
        )
    
    return queue


# ============================================================================
# Type Aliases for Cleaner Signatures
# ============================================================================

RequestID = Annotated[str, Depends(get_request_id)]
APIKey = Annotated[APIKeyData, Depends(get_api_key)]
Context = Annotated[RequestContext, Depends(get_request_context)]
Queue = Annotated[AudioRAGQueue, Depends(get_queue)]
