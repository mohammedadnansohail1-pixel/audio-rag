"""Middleware for error handling, logging, and request context."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from audio_rag.queue.exceptions import (
    QueueError,
    DuplicateJobError,
    JobNotFoundError,
    QueueFullError,
    RedisConnectionError,
    InvalidAudioError,
    TenantValidationError,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Request ID Middleware
# ============================================================================

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add X-Request-ID header to all requests/responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get or generate request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        
        # Store in request state
        request.state.request_id = request_id
        
        # Process request
        response = await call_next(request)
        
        # Add to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response


# ============================================================================
# Logging Middleware
# ============================================================================

class LoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with timing information."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()
        
        # Get request info
        request_id = getattr(request.state, "request_id", "unknown")
        method = request.method
        path = request.url.path
        client_ip = request.client.host if request.client else "unknown"
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Log based on status code
        status_code = response.status_code
        log_msg = (
            f"{method} {path} - {status_code} - {duration_ms:.1f}ms "
            f"[{request_id}] from {client_ip}"
        )
        
        if status_code >= 500:
            logger.error(log_msg)
        elif status_code >= 400:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
        
        # Add timing header
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.1f}"
        
        return response


# ============================================================================
# Rate Limit Headers Middleware
# ============================================================================

class RateLimitHeadersMiddleware(BaseHTTPMiddleware):
    """Add rate limit headers to responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Check if rate limit info was stored
        rate_limit = getattr(request.state, "rate_limit", None)
        if rate_limit:
            response.headers["X-RateLimit-Limit"] = str(rate_limit.limit)
            response.headers["X-RateLimit-Remaining"] = str(rate_limit.remaining)
            response.headers["X-RateLimit-Reset"] = str(rate_limit.reset_at)
        
        return response


# ============================================================================
# Exception Handlers
# ============================================================================

def create_error_response(
    status_code: int,
    error: str,
    code: str,
    request_id: str | None = None,
    details: dict | None = None,
    retry_after: int | None = None,
) -> JSONResponse:
    """Create consistent error response."""
    content = {
        "error": error,
        "code": code,
        "status": status_code,
    }
    
    if request_id:
        content["request_id"] = request_id
    if details:
        content["details"] = details
    if retry_after:
        content["retry_after"] = retry_after
    
    headers = {}
    if retry_after:
        headers["Retry-After"] = str(retry_after)
    
    return JSONResponse(
        status_code=status_code,
        content=content,
        headers=headers if headers else None,
    )


async def queue_error_handler(request: Request, exc: QueueError) -> JSONResponse:
    """Handle queue-related errors."""
    request_id = getattr(request.state, "request_id", None)
    
    # Map exception types to HTTP status codes
    if isinstance(exc, DuplicateJobError):
        return create_error_response(
            status_code=409,
            error="Job already exists",
            code="DUPLICATE_JOB",
            request_id=request_id,
            details={"existing_job_id": exc.existing_job_id},
        )
    
    if isinstance(exc, JobNotFoundError):
        return create_error_response(
            status_code=404,
            error="Job not found",
            code="JOB_NOT_FOUND",
            request_id=request_id,
            details={"job_id": exc.job_id},
        )
    
    if isinstance(exc, QueueFullError):
        return create_error_response(
            status_code=503,
            error="Queue is full, try again later",
            code="QUEUE_FULL",
            request_id=request_id,
            retry_after=60,
            details={
                "queue": exc.queue_name,
                "depth": exc.current_depth,
                "max_depth": exc.max_depth,
            },
        )
    
    if isinstance(exc, RedisConnectionError):
        logger.error(f"Redis connection error: {exc}")
        return create_error_response(
            status_code=503,
            error="Service temporarily unavailable",
            code="SERVICE_UNAVAILABLE",
            request_id=request_id,
            retry_after=30,
        )
    
    if isinstance(exc, InvalidAudioError):
        return create_error_response(
            status_code=422,
            error=str(exc),
            code="INVALID_AUDIO",
            request_id=request_id,
            details={"reason": exc.reason},
        )
    
    if isinstance(exc, TenantValidationError):
        return create_error_response(
            status_code=400,
            error=str(exc),
            code="INVALID_TENANT",
            request_id=request_id,
            details={"reason": exc.reason},
        )
    
    # Generic queue error
    logger.error(f"Queue error: {exc}")
    return create_error_response(
        status_code=500 if not exc.recoverable else 503,
        error="Queue operation failed",
        code="QUEUE_ERROR",
        request_id=request_id,
        retry_after=30 if exc.recoverable else None,
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    request_id = getattr(request.state, "request_id", None)
    
    logger.exception(f"Unhandled exception [{request_id}]: {exc}")
    
    return create_error_response(
        status_code=500,
        error="Internal server error",
        code="INTERNAL_ERROR",
        request_id=request_id,
    )


# ============================================================================
# Setup Function
# ============================================================================

def setup_middleware(app: FastAPI) -> None:
    """Add all middleware to the app.
    
    Order matters! Middleware is executed in reverse order of addition.
    Last added = first executed on request, last executed on response.
    """
    # Exception handlers
    app.add_exception_handler(QueueError, queue_error_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
    
    # Middleware (reverse order of execution)
    app.add_middleware(RateLimitHeadersMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)
