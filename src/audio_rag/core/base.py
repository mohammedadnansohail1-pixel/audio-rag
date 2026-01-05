"""Base classes for Audio RAG components."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TranscriptSegment:
    """A segment of transcribed audio with timing."""
    text: str
    start: float
    end: float
    confidence: float = 1.0
    words: list[dict] | None = None
    speaker: str | None = None
    language: str | None = None


@dataclass
class AudioChunk:
    """A chunk of audio transcript ready for embedding."""
    text: str
    start: float
    end: float
    speaker: str | None = None
    metadata: dict | None = None


@dataclass
class SparseVector:
    """Sparse vector representation for BM25/lexical search."""
    indices: list[int]
    values: list[float]
    
    def to_dict(self) -> dict[int, float]:
        return dict(zip(self.indices, self.values))


@dataclass
class EmbeddingResult:
    """Result of embedding containing dense and optional sparse vectors."""
    dense: list[float]
    sparse: SparseVector | None = None


@dataclass
class RetrievalResult:
    """A retrieved chunk with relevance score."""
    chunk: AudioChunk
    score: float
    source: str | None = None


class BaseASR(ABC):
    """Abstract base class for ASR backends."""

    @abstractmethod
    def transcribe(self, audio_path: str) -> list[TranscriptSegment]:
        pass

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def unload(self) -> None:
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        pass

    @property
    @abstractmethod
    def vram_required(self) -> float:
        pass


class BaseDiarizer(ABC):
    """Abstract base class for speaker diarization backends."""

    @abstractmethod
    def diarize(self, audio_path: str) -> list[dict]:
        pass

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def unload(self) -> None:
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        pass

    @property
    @abstractmethod
    def vram_required(self) -> float:
        pass


class BaseChunker(ABC):
    """Abstract base class for text chunking strategies."""

    @abstractmethod
    def chunk(
        self,
        segments: list[TranscriptSegment],
        speaker_segments: list[dict] | None = None,
    ) -> list[AudioChunk]:
        pass


class BaseEmbedder(ABC):
    """Abstract base class for embedding backends."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate embeddings for texts (dense + optional sparse)."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> EmbeddingResult:
        """Generate embedding for a query (dense + optional sparse)."""
        pass

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def unload(self) -> None:
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        pass

    @property
    @abstractmethod
    def vram_required(self) -> float:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @property
    def supports_sparse(self) -> bool:
        """Whether this embedder supports sparse vectors."""
        return False


class BaseRetriever(ABC):
    """Abstract base class for retrieval backends."""

    @abstractmethod
    def add(
        self,
        chunks: list[AudioChunk],
        embeddings: list[EmbeddingResult],
        collection_name: str | None = None,
    ) -> None:
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: EmbeddingResult,
        top_k: int | None = None,
        collection_name: str | None = None,
        filter_metadata: dict | None = None,
    ) -> list[RetrievalResult]:
        pass


class BaseTTS(ABC):
    """Abstract base class for TTS backends."""

    @abstractmethod
    def synthesize(self, text: str, output_path: str) -> str:
        pass

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def unload(self) -> None:
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        pass

    @property
    @abstractmethod
    def vram_required(self) -> float:
        pass
