"""API configuration with Pydantic models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RateLimitConfig(BaseModel):
    """Rate limit settings for a single endpoint type."""
    
    requests: int = Field(ge=1, description="Max requests")
    window_seconds: int = Field(ge=1, description="Time window")


class TierRateLimits(BaseModel):
    """Rate limits for a user tier."""
    
    query: RateLimitConfig = RateLimitConfig(requests=10, window_seconds=60)
    ingest: RateLimitConfig = RateLimitConfig(requests=5, window_seconds=3600)
    status: RateLimitConfig = RateLimitConfig(requests=60, window_seconds=60)


class RateLimitSettings(BaseModel):
    """Rate limit settings per tier."""
    
    free: TierRateLimits = TierRateLimits()
    basic: TierRateLimits = TierRateLimits(
        query=RateLimitConfig(requests=60, window_seconds=60),
        ingest=RateLimitConfig(requests=20, window_seconds=3600),
        status=RateLimitConfig(requests=120, window_seconds=60),
    )
    premium: TierRateLimits = TierRateLimits(
        query=RateLimitConfig(requests=300, window_seconds=60),
        ingest=RateLimitConfig(requests=100, window_seconds=3600),
        status=RateLimitConfig(requests=600, window_seconds=60),
    )
    
    def get_tier(self, tier: str) -> TierRateLimits:
        """Get rate limits for a tier."""
        return getattr(self, tier, self.free)


class TimeoutSettings(BaseModel):
    """Timeout settings for different operations."""
    
    health: float = 1.0
    status: float = 5.0
    query: float = 30.0
    ingest_accept: float = 10.0  # Just enqueueing
    file_upload: float = 300.0  # Large file upload


class UploadSettings(BaseModel):
    """File upload settings."""
    
    max_size_mb: float = Field(default=500.0, ge=1, le=2000)
    max_duration_minutes: float = Field(default=180.0, ge=1)
    allowed_extensions: set[str] = {
        ".mp3", ".wav", ".flac", ".ogg", 
        ".m4a", ".aac", ".opus", ".webm",
    }
    chunk_size: int = Field(default=1024 * 1024, description="1MB chunks")
    
    @property
    def max_size_bytes(self) -> int:
        return int(self.max_size_mb * 1024 * 1024)


class AuthSettings(BaseModel):
    """Authentication settings."""
    
    header_name: str = "X-API-Key"
    # In production, load valid keys from DB/Redis
    # This is just for development
    dev_api_keys: dict[str, dict] = {
        "dev-key-12345": {
            "tenant_id": "audio_rag_unt_cs_5500_fall2025",
            "tier": "premium",
            "name": "Development Key",
        },
    }


class APIConfig(BaseModel):
    """Complete API configuration."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # Feature flags
    enable_docs: bool = True
    enable_cors: bool = True
    cors_origins: list[str] = ["*"]
    
    # Sub-configs
    rate_limits: RateLimitSettings = RateLimitSettings()
    timeouts: TimeoutSettings = TimeoutSettings()
    upload: UploadSettings = UploadSettings()
    auth: AuthSettings = AuthSettings()
    
    # Redis (for rate limiting)
    redis_url: str = "redis://localhost:6379/0"


# Default configuration
DEFAULT_API_CONFIG = APIConfig()
