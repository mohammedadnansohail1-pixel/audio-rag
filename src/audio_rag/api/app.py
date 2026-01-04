"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from audio_rag.api.config import APIConfig, DEFAULT_API_CONFIG
from audio_rag.api.health import router as health_router
from audio_rag.api.middleware import setup_middleware

if TYPE_CHECKING:
    from audio_rag.queue import AudioRAGQueue

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    logger.info("Starting Audio RAG API...")
    
    try:
        from audio_rag.queue import AudioRAGQueue, DEFAULT_QUEUE_CONFIG
        from audio_rag.queue.connection import RedisConnectionManager
        from audio_rag.queue.validation import DEFAULT_JOB_VALIDATOR
        
        # Create connection manager first
        conn_manager = RedisConnectionManager.from_queue_config(DEFAULT_QUEUE_CONFIG)
        
        # Create queue with connection manager
        queue = AudioRAGQueue(
            connection_manager=conn_manager,
            config=DEFAULT_QUEUE_CONFIG,
            validator=DEFAULT_JOB_VALIDATOR,
        )
        app.state.queue = queue
        app.state.queue_healthy = queue.is_healthy()
        
        if app.state.queue_healthy:
            logger.info("Queue connection established")
        else:
            logger.warning("Queue unhealthy at startup")
            
    except Exception as e:
        logger.error(f"Failed to initialize queue: {e}")
        app.state.queue = None
        app.state.queue_healthy = False
    
    app.state.initialized = True
    logger.info("Audio RAG API started")
    
    yield
    
    logger.info("Shutting down Audio RAG API...")
    
    if hasattr(app.state, 'queue') and app.state.queue:
        try:
            app.state.queue.close()
        except Exception as e:
            logger.error(f"Error closing queue: {e}")
    
    if hasattr(app.state, 'embedder') and app.state.embedder:
        try:
            del app.state.embedder
        except Exception as e:
            logger.error(f"Error closing embedder: {e}")
    
    if hasattr(app.state, 'qdrant_client') and app.state.qdrant_client:
        try:
            app.state.qdrant_client.close()
        except Exception as e:
            logger.error(f"Error closing Qdrant client: {e}")
    
    logger.info("Audio RAG API shutdown complete")


def create_app(config: APIConfig | None = None) -> FastAPI:
    """Create and configure FastAPI application."""
    config = config or DEFAULT_API_CONFIG
    
    app = FastAPI(
        title="Audio RAG API",
        description="Audio transcription, indexing, and semantic search API",
        version="1.0.0",
        docs_url="/docs" if config.enable_docs else None,
        redoc_url="/redoc" if config.enable_docs else None,
        openapi_url="/openapi.json" if config.enable_docs else None,
        lifespan=lifespan,
    )
    
    app.state.config = config
    app.state.initialized = False
    
    if config.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    setup_middleware(app)
    
    app.include_router(health_router, prefix="/health", tags=["health"])
    
    from audio_rag.api.v1.router import router as v1_router
    app.include_router(v1_router, prefix="/api/v1")
    
    return app


app = create_app()
