"""Configuration management."""

from audio_rag.config.schema import (
    AudioRAGConfig,
    ASRConfig,
    DiarizationConfig,
    AlignmentConfig,
    ChunkingConfig,
    EmbeddingConfig,
    RetrievalConfig,
    TTSConfig,
    ResourceConfig,
)
from audio_rag.config.loader import load_config, load_yaml, deep_merge

__all__ = [
    # Main config
    "AudioRAGConfig",
    "load_config",
    # Sub-configs
    "ASRConfig",
    "DiarizationConfig",
    "AlignmentConfig",
    "ChunkingConfig",
    "EmbeddingConfig",
    "RetrievalConfig",
    "TTSConfig",
    "ResourceConfig",
    # Utilities
    "load_yaml",
    "deep_merge",
]
