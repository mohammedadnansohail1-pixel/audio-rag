"""Pydantic configuration schemas with validation."""

from typing import Literal
from pydantic import BaseModel, Field


class ASRConfig(BaseModel):
    """ASR (Automatic Speech Recognition) configuration."""
    backend: Literal["faster-whisper"] = "faster-whisper"
    model_size: Literal["tiny", "base", "small", "medium", "large-v2", "large-v3"] = "large-v3"
    device: Literal["cuda", "cpu", "auto"] = "auto"
    compute_type: Literal["float16", "int8", "float32"] = "float16"
    vad_filter: bool = True
    vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    language: str | None = None  # None = auto-detect


class DiarizationConfig(BaseModel):
    """Speaker diarization configuration."""
    backend: Literal["pyannote"] = "pyannote"
    model: str = "pyannote/speaker-diarization-3.1"
    device: Literal["cuda", "cpu", "auto"] = "auto"
    min_speakers: int | None = None
    max_speakers: int | None = None
    min_speech_duration_ms: int = Field(default=250, ge=0)


class AlignmentConfig(BaseModel):
    """Transcript-diarization alignment configuration."""
    method: Literal["word_level", "segment_level"] = "word_level"
    use_whisperx: bool = True  # For word-level timestamps


class ChunkingConfig(BaseModel):
    """Chunking strategy configuration."""
    strategy: Literal["speaker_turn", "semantic", "fixed"] = "speaker_turn"
    max_tokens: int = Field(default=500, ge=50, le=2000)
    overlap_tokens: int = Field(default=50, ge=0)
    min_chunk_tokens: int = Field(default=50, ge=1)  # Allow 1 for testing


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    backend: Literal["bge-m3", "multilingual-e5"] = "bge-m3"
    model: str = "BAAI/bge-m3"
    device: Literal["cuda", "cpu", "auto"] = "auto"
    batch_size: int = Field(default=32, ge=1)
    normalize: bool = True


class RetrievalConfig(BaseModel):
    """Vector retrieval configuration."""
    backend: Literal["qdrant"] = "qdrant"
    collection_name: str = "audio_rag"
    search_type: Literal["dense", "sparse", "hybrid"] = "hybrid"
    top_k: int = Field(default=5, ge=1, le=100)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Qdrant connection
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_in_memory: bool = False  # True for testing


class TTSConfig(BaseModel):
    """Text-to-speech configuration."""
    backend: Literal["piper", "edge-tts"] = "piper"
    model: str = "en_US-lessac-medium"
    fallback_backend: Literal["edge-tts", "none"] = "edge-tts"
    output_format: Literal["wav", "mp3"] = "wav"
    sample_rate: int = Field(default=22050, ge=8000, le=48000)


class ResourceConfig(BaseModel):
    """Resource management configuration."""
    max_vram_gb: float = Field(default=12.0, ge=1.0)
    max_ram_gb: float = Field(default=16.0, ge=1.0)
    unload_after_idle_seconds: int = Field(default=300, ge=0)
    subprocess_isolation: bool = True
    max_audio_duration_minutes: int = Field(default=30, ge=1)


class AudioRAGConfig(BaseModel):
    """Root configuration for Audio RAG system."""
    asr: ASRConfig = Field(default_factory=ASRConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    
    # Global settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    data_dir: str = "./data"
    cache_dir: str = "./cache"
