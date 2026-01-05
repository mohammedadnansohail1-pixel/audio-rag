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
    language: str | None = None


class DiarizationConfig(BaseModel):
    """Speaker diarization configuration."""
    backend: Literal["pyannote", "nemo"] = "nemo"
    model: str = "pyannote/speaker-diarization-3.1"
    device: Literal["cuda", "cpu", "auto"] = "auto"
    min_speakers: int | None = None
    max_speakers: int | None = None
    min_speech_duration_ms: int = Field(default=250, ge=0)


class AlignmentConfig(BaseModel):
    """Transcript-diarization alignment configuration."""
    method: Literal["word_level", "segment_level"] = "word_level"
    use_whisperx: bool = True


class ChunkingConfig(BaseModel):
    """Chunking strategy configuration."""
    strategy: Literal["speaker_turn", "semantic", "fixed"] = "speaker_turn"
    max_tokens: int = Field(default=256, ge=50, le=2000)
    overlap_tokens: int = Field(default=50, ge=0)
    min_chunk_tokens: int = Field(default=30, ge=1)


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    backend: Literal["bge-m3", "multilingual-e5"] = "bge-m3"
    model: str = "BAAI/bge-m3"
    device: Literal["cuda", "cpu", "auto"] = "auto"
    batch_size: int = Field(default=32, ge=1)
    normalize: bool = True
    use_sparse: bool = True  # Enable sparse vectors for hybrid search


class RetrievalConfig(BaseModel):
    """Vector retrieval configuration."""
    backend: Literal["qdrant"] = "qdrant"
    collection_name: str = "audio_rag"
    search_type: Literal["dense", "sparse", "hybrid"] = "hybrid"
    top_k: int = Field(default=5, ge=1, le=100)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_in_memory: bool = False
    # Hybrid search weights (must sum to 1.0)
    dense_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    sparse_weight: float = Field(default=0.3, ge=0.0, le=1.0)


class RerankingConfig(BaseModel):
    """Reranking configuration for improving retrieval accuracy."""
    backend: Literal["bge-reranker", "none"] = "bge-reranker"
    model: str = "BAAI/bge-reranker-base"
    device: Literal["cuda", "cpu", "auto"] = "auto"
    top_k: int = Field(default=5, ge=1, le=50)
    initial_k: int = Field(default=20, ge=1, le=100)
    batch_size: int = Field(default=16, ge=1)


class GenerationConfig(BaseModel):
    """LLM answer generation configuration."""
    backend: Literal["ollama", "none"] = "ollama"
    model: str = "llama3.2:3b"
    base_url: str = "http://localhost:11434"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    timeout: float = Field(default=60.0, ge=1.0)
    fallback_models: list[str] = Field(default_factory=lambda: ["llama3.1:8b", "mistral:7b"])


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
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    data_dir: str = "./data"
    cache_dir: str = "./cache"
