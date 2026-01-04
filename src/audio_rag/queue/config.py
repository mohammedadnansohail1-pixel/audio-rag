"""Queue configuration schema.

Defines Pydantic models for queue configuration with validation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class QueueDefinition(BaseModel):
    """Configuration for a single queue."""
    
    name: str = Field(..., description="Queue name (e.g., 'high', 'normal', 'low')")
    timeout: int = Field(
        default=1800,
        ge=60,
        le=7200,
        description="Job timeout in seconds (1 min - 2 hours)",
    )
    max_depth: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum number of jobs in queue",
    )
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure queue name is valid."""
        if not v.isalnum() and "_" not in v:
            raise ValueError("Queue name must be alphanumeric (underscores allowed)")
        return v.lower()


class WorkerConfig(BaseModel):
    """Configuration for GPU worker."""
    
    max_memory_mb: int = Field(
        default=10000,
        ge=1000,
        le=100000,
        description="Maximum GPU memory budget in MB",
    )
    poll_interval: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Queue poll interval in seconds",
    )
    job_timeout: int = Field(
        default=1800,
        ge=60,
        le=7200,
        description="Default job timeout in seconds",
    )
    heartbeat_interval: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Worker heartbeat interval in seconds",
    )
    shutdown_timeout: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Graceful shutdown timeout in seconds",
    )
    max_consecutive_failures: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max consecutive job failures before worker pauses",
    )
    preload_models: bool = Field(
        default=True,
        description="Preload ML models on worker startup",
    )
    queue_prefix: str = Field(
        default="audio_rag",
        description="Prefix for queue names",
    )


class RedisConfig(BaseModel):
    """Redis connection configuration."""
    
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, ge=1, le=65535, description="Redis port")
    db: int = Field(default=0, ge=0, le=15, description="Redis database number")
    password: str | None = Field(default=None, description="Redis password")
    socket_timeout: float = Field(
        default=5.0,
        ge=1.0,
        le=60.0,
        description="Socket timeout in seconds",
    )
    socket_connect_timeout: float = Field(
        default=5.0,
        ge=1.0,
        le=30.0,
        description="Connection timeout in seconds",
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max connection retry attempts",
    )
    retry_backoff: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Base backoff time between retries in seconds",
    )
    
    @property
    def url(self) -> str:
        """Build Redis URL from components."""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class QueueConfig(BaseModel):
    """Main queue configuration."""
    
    redis: RedisConfig = Field(default_factory=RedisConfig)
    queues: list[QueueDefinition] = Field(
        default_factory=lambda: [
            QueueDefinition(name="high", timeout=3600, max_depth=50),
            QueueDefinition(name="normal", timeout=1800, max_depth=100),
            QueueDefinition(name="low", timeout=3600, max_depth=200),
        ],
        description="Queue definitions in priority order (highest first)",
    )
    worker: WorkerConfig = Field(default_factory=WorkerConfig)
    
    # Idempotency settings
    idempotency_ttl: int = Field(
        default=86400,
        ge=3600,
        le=604800,
        description="Idempotency key TTL in seconds (1 hour - 1 week)",
    )
    
    # Job result settings
    result_ttl: int = Field(
        default=86400,
        ge=3600,
        le=604800,
        description="Job result TTL in seconds",
    )
    
    # Checkpoint settings
    checkpoint_ttl: int = Field(
        default=86400,
        ge=3600,
        le=604800,
        description="Checkpoint TTL in seconds",
    )
    
    @field_validator("queues")
    @classmethod
    def validate_queues(cls, v: list[QueueDefinition]) -> list[QueueDefinition]:
        """Ensure at least one queue and no duplicate names."""
        if not v:
            raise ValueError("At least one queue must be defined")
        
        names = [q.name for q in v]
        if len(names) != len(set(names)):
            raise ValueError("Queue names must be unique")
        
        return v
    
    def get_queue(self, name: str) -> QueueDefinition | None:
        """Get queue definition by name."""
        for q in self.queues:
            if q.name == name:
                return q
        return None
    
    def get_queue_names(self) -> list[str]:
        """Get list of queue names in priority order."""
        return [q.name for q in self.queues]
    
    def get_timeout_for_queue(self, name: str) -> int:
        """Get timeout for a specific queue."""
        queue = self.get_queue(name)
        return queue.timeout if queue else self.worker.job_timeout


# Default configuration instance
DEFAULT_QUEUE_CONFIG = QueueConfig()
