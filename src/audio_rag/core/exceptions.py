"""Custom exceptions for Audio RAG system."""


class AudioRAGError(Exception):
    """Base exception for all Audio RAG errors."""
    pass


class ConfigError(AudioRAGError):
    """Configuration loading or validation error."""
    pass


class RegistryError(AudioRAGError):
    """Component registry error."""
    pass


class ResourceError(AudioRAGError):
    """Resource management error (VRAM, memory, etc.)."""
    pass


class ASRError(AudioRAGError):
    """Speech recognition error."""
    pass


class DiarizationError(AudioRAGError):
    """Speaker diarization error."""
    pass


class AlignmentError(AudioRAGError):
    """Transcript-diarization alignment error."""
    pass


class EmbeddingError(AudioRAGError):
    """Embedding generation error."""
    pass


class RetrievalError(AudioRAGError):
    """Vector retrieval error."""
    pass


class TTSError(AudioRAGError):
    """Text-to-speech error."""
    pass


class PipelineError(AudioRAGError):
    """Pipeline orchestration error."""
    pass
