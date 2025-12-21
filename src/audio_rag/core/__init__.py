"""Core components: registry, base classes, exceptions."""

from audio_rag.core.registry import Registry
from audio_rag.core.base import (
    TranscriptSegment,
    AudioChunk,
    RetrievalResult,
    BaseASR,
    BaseDiarizer,
    BaseChunker,
    BaseEmbedder,
    BaseRetriever,
    BaseTTS,
)
from audio_rag.core.exceptions import (
    AudioRAGError,
    ConfigError,
    RegistryError,
    ResourceError,
    ASRError,
    DiarizationError,
    AlignmentError,
    EmbeddingError,
    RetrievalError,
    TTSError,
    PipelineError,
)

__all__ = [
    # Registry
    "Registry",
    # Data classes
    "TranscriptSegment",
    "AudioChunk",
    "RetrievalResult",
    # Base classes
    "BaseASR",
    "BaseDiarizer",
    "BaseChunker",
    "BaseEmbedder",
    "BaseRetriever",
    "BaseTTS",
    # Exceptions
    "AudioRAGError",
    "ConfigError",
    "RegistryError",
    "ResourceError",
    "ASRError",
    "DiarizationError",
    "AlignmentError",
    "EmbeddingError",
    "RetrievalError",
    "TTSError",
    "PipelineError",
]
