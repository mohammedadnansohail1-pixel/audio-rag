"""BGE Reranker implementation using CrossEncoder."""

import torch
from sentence_transformers import CrossEncoder

from audio_rag.reranking.base import BaseReranker, RerankerRegistry
from audio_rag.config import RerankingConfig
from audio_rag.core import RetrievalResult, RerankingError
from audio_rag.utils import get_logger

logger = get_logger(__name__)


@RerankerRegistry.register("bge-reranker")
class BGEReranker(BaseReranker):
    """BGE Reranker using CrossEncoder architecture.
    
    Supports:
    - BAAI/bge-reranker-base (~400MB, fast)
    - BAAI/bge-reranker-v2-m3 (~560MB, multilingual, 8k context)
    """

    MODEL_VRAM = {
        "BAAI/bge-reranker-base": 0.5,
        "BAAI/bge-reranker-large": 1.2,
        "BAAI/bge-reranker-v2-m3": 1.5,
    }

    def __init__(self, config: RerankingConfig):
        super().__init__(config)
        self._model: CrossEncoder | None = None
        self._device: str | None = None

    @property
    def vram_required(self) -> float:
        return self.MODEL_VRAM.get(self.config.model, 1.0)

    def _resolve_device(self) -> str:
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device

    def load(self) -> None:
        if self._is_loaded:
            return

        self._device = self._resolve_device()
        logger.info(f"Loading reranker {self.config.model} on {self._device}")

        try:
            self._model = CrossEncoder(
                self.config.model,
                max_length=512,
                device=self._device,
            )
            self._is_loaded = True
            logger.info(f"Reranker loaded: {self.config.model}")
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
            # Try CPU fallback if GPU failed
            if self._device == "cuda":
                logger.warning("Retrying on CPU...")
                try:
                    self._device = "cpu"
                    self._model = CrossEncoder(
                        self.config.model,
                        max_length=512,
                        device="cpu",
                    )
                    self._is_loaded = True
                    logger.info(f"Reranker loaded on CPU fallback")
                except Exception as e2:
                    raise RerankingError(f"Failed to load reranker: {e2}") from e2
            else:
                raise RerankingError(f"Failed to load reranker: {e}") from e

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._is_loaded = False
        logger.info("Reranker unloaded")

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank results using cross-encoder scoring.
        
        Args:
            query: User query
            results: Initial retrieval results from vector search
            top_k: Number of top results to return (default: config.top_k)
            
        Returns:
            Reranked results sorted by cross-encoder score
        """
        if not results:
            return results

        top_k = top_k or self.config.top_k

        # If fewer results than top_k, just return sorted by original score
        if len(results) <= top_k:
            return sorted(results, key=lambda x: x.score, reverse=True)

        try:
            if not self._is_loaded:
                self.load()

            # Build query-document pairs
            pairs = [(query, r.chunk.text) for r in results]

            # Get cross-encoder scores
            scores = self._model.predict(
                pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
            )

            # Create new results with reranked scores
            reranked = []
            for i, result in enumerate(results):
                reranked.append(RetrievalResult(
                    chunk=result.chunk,
                    score=float(scores[i]),  # Replace with reranker score
                ))

            # Sort by new score and take top_k
            reranked.sort(key=lambda x: x.score, reverse=True)
            
            logger.debug(
                f"Reranked {len(results)} results, "
                f"top score: {reranked[0].score:.3f} -> {reranked[-1].score:.3f}"
            )
            
            return reranked[:top_k]

        except Exception as e:
            logger.warning(f"Reranking failed: {e}, returning original top-{top_k}")
            # Graceful degradation: return original results sorted by score
            sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
            return sorted_results[:top_k]
