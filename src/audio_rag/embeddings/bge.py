"""BGE-M3 multilingual embedding implementation."""

import gc
from audio_rag.embeddings.base import EmbeddingsRegistry
from audio_rag.core import BaseEmbedder, EmbeddingError
from audio_rag.config import EmbeddingConfig
from audio_rag.utils import get_logger, timed, require_loaded

logger = get_logger(__name__)

# VRAM estimate for BGE-M3
VRAM_ESTIMATE = 2.5  # GB


@EmbeddingsRegistry.register("bge-m3")
class BGEM3Embedder(BaseEmbedder):
    """BGE-M3 multilingual embedding backend.
    
    Supports dense, sparse, and multi-vector embeddings.
    Excellent multilingual support (100+ languages).
    """
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._model = None
        self._device = self._resolve_device(config.device)
        self._dimension = 1024  # BGE-M3 dimension
        logger.info(f"BGEM3Embedder initialized: model={config.model}, device={self._device}")
    
    def _resolve_device(self, device: str) -> str:
        """Resolve 'auto' to actual device."""
        if device != "auto":
            return device
        
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    
    def load(self) -> None:
        """Load embedding model into memory."""
        if self._model is not None:
            logger.debug("Model already loaded")
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading {self.config.model} on {self._device}...")
            self._model = SentenceTransformer(
                self.config.model,
                device=self._device,
            )
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded (dim={self._dimension})")
            
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model: {e}")
    
    def unload(self) -> None:
        """Unload model and free memory."""
        if self._model is None:
            return
        
        logger.info("Unloading embedding model...")
        del self._model
        self._model = None
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        logger.info("Embedding model unloaded")
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
    
    @property
    def vram_required(self) -> float:
        """VRAM required in GB."""
        return VRAM_ESTIMATE
    
    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return self._dimension
    
    @timed
    @require_loaded
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            embeddings = self._model.encode(
                texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize,
                show_progress_bar=len(texts) > 10,
            )
            
            logger.debug(f"Generated {len(embeddings)} embeddings")
            return embeddings.tolist()
            
        except Exception as e:
            raise EmbeddingError(f"Embedding generation failed: {e}")
    
    @require_loaded
    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        try:
            embedding = self._model.encode(
                query,
                normalize_embeddings=self.config.normalize,
            )
            return embedding.tolist()
            
        except Exception as e:
            raise EmbeddingError(f"Query embedding failed: {e}")
