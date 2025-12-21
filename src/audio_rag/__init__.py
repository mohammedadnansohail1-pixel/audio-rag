"""Audio RAG - Production-grade, config-driven Audio RAG system.

Usage:
    from audio_rag import AudioRAG
    
    # Initialize from config
    rag = AudioRAG.from_config(env="development")
    
    # Ingest audio
    result = rag.ingest("podcast.mp3")
    
    # Query
    response = rag.query("What was discussed about AI?")
"""

from audio_rag.pipeline import AudioRAG, IngestionResult, QueryResult
from audio_rag.config import AudioRAGConfig, load_config

__version__ = "0.1.0"

__all__ = [
    "AudioRAG",
    "AudioRAGConfig",
    "IngestionResult",
    "QueryResult",
    "load_config",
]
