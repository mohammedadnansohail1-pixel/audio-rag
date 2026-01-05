"""Base class and registry for rerankers."""

from abc import ABC, abstractmethod
from typing import Any

from audio_rag.config import RerankingConfig
from audio_rag.core import RetrievalResult
from audio_rag.utils import get_logger

logger = get_logger(__name__)


class BaseReranker(ABC):
    """Base class for rerankers."""

    def __init__(self, config: RerankingConfig):
        self.config = config
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    @abstractmethod
    def vram_required(self) -> float:
        """VRAM required in GB."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load model into memory."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload model from memory."""
        pass

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank retrieval results.
        
        Args:
            query: User query
            results: Initial retrieval results
            top_k: Number of results to return after reranking
            
        Returns:
            Reranked results with updated scores
        """
        pass


class RerankerRegistry:
    """Registry for reranker backends."""

    _rerankers: dict[str, type[BaseReranker]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(reranker_cls: type[BaseReranker]) -> type[BaseReranker]:
            cls._rerankers[name] = reranker_cls
            logger.debug(f"Registered reranker: {name}")
            return reranker_cls
        return decorator

    @classmethod
    def create(cls, name: str, config: RerankingConfig) -> BaseReranker | None:
        if name == "none":
            return None
        if name not in cls._rerankers:
            available = list(cls._rerankers.keys())
            raise ValueError(f"Unknown reranker: {name}. Available: {available}")
        return cls._rerankers[name](config)

    @classmethod
    def list_available(cls) -> list[str]:
        return list(cls._rerankers.keys())
