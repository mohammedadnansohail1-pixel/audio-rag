"""Abstract base classes defining component interfaces."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TranscriptSegment:
    """A segment of transcribed audio with timing and speaker info."""
    text: str
    start: float  # seconds
    end: float  # seconds
    speaker: str | None = None
    confidence: float | None = None
    language: str | None = None


@dataclass
class AudioChunk:
    """A chunk of transcript ready for embedding."""
    text: str
    start: float
    end: float
    speaker: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class RetrievalResult:
    """A retrieved chunk with relevance score."""
    chunk: AudioChunk
    score: float
    source: str | None = None


class BaseASR(ABC):
    """Abstract base class for ASR (Automatic Speech Recognition) backends."""
    
    @abstractmethod
    def transcribe(self, audio_path: Path, language: str | None = None) -> list[TranscriptSegment]:
        """Transcribe audio file to segments."""
        pass
    
    @abstractmethod
    def load(self) -> None:
        """Load model into memory."""
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Unload model from memory."""
        pass
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass
    
    @property
    @abstractmethod
    def vram_required(self) -> float:
        """VRAM required in GB."""
        pass


class BaseDiarizer(ABC):
    """Abstract base class for speaker diarization backends."""
    
    @abstractmethod
    def diarize(
        self, audio_path: Path, min_speakers: int | None = None, max_speakers: int | None = None
    ) -> list[TranscriptSegment]:
        """Identify speaker segments in audio."""
        pass
    
    @abstractmethod
    def load(self) -> None:
        """Load model into memory."""
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Unload model from memory."""
        pass
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass
    
    @property
    @abstractmethod
    def vram_required(self) -> float:
        """VRAM required in GB."""
        pass


class BaseChunker(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    def chunk(self, segments: list[TranscriptSegment]) -> list[AudioChunk]:
        """Convert transcript segments into chunks for embedding."""
        pass


class BaseEmbedder(ABC):
    """Abstract base class for embedding backends."""
    
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts."""
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a query."""
        pass
    
    @abstractmethod
    def load(self) -> None:
        """Load model into memory."""
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Unload model from memory."""
        pass
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass
    
    @property
    @abstractmethod
    def vram_required(self) -> float:
        """VRAM required in GB."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension."""
        pass


class BaseRetriever(ABC):
    """Abstract base class for retrieval backends."""
    
    @abstractmethod
    def add(self, chunks: list[AudioChunk], embeddings: list[list[float]]) -> None:
        """Add chunks with their embeddings to the store."""
        pass
    
    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievalResult]:
        """Search for similar chunks."""
        pass
    
    @abstractmethod
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        pass


class BaseTTS(ABC):
    """Abstract base class for TTS (Text-to-Speech) backends."""
    
    @abstractmethod
    def synthesize(self, text: str, output_path: Path, language: str | None = None) -> Path:
        """Synthesize text to audio file."""
        pass
    
    @abstractmethod
    def load(self) -> None:
        """Load model into memory."""
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Unload model from memory."""
        pass
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass
    
    @property
    @abstractmethod
    def vram_required(self) -> float:
        """VRAM required in GB."""
        pass
