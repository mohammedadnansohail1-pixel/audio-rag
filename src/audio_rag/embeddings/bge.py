"""BGE-M3 multilingual embedding implementation with sparse vector support."""

import gc
from audio_rag.embeddings.base import EmbeddingsRegistry
from audio_rag.core import BaseEmbedder, EmbeddingError, EmbeddingResult, SparseVector
from audio_rag.config import EmbeddingConfig
from audio_rag.utils import get_logger, timed, require_loaded

logger = get_logger(__name__)

VRAM_ESTIMATE = 2.5  # GB


@EmbeddingsRegistry.register("bge-m3")
class BGEM3Embedder(BaseEmbedder):
    """BGE-M3 multilingual embedding backend.

    Supports dense and sparse (lexical) embeddings for hybrid search.
    Uses FlagEmbedding for sparse vector generation.
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._model = None
        self._device = self._resolve_device(config.device)
        self._dimension = 1024  # BGE-M3 dimension
        self._use_sparse = config.use_sparse
        logger.info(
            f"BGEM3Embedder initialized: model={config.model}, "
            f"device={self._device}, sparse={self._use_sparse}"
        )

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def load(self) -> None:
        if self._model is not None:
            logger.debug("Model already loaded")
            return

        try:
            from FlagEmbedding import BGEM3FlagModel

            logger.info(f"Loading {self.config.model} on {self._device}...")
            self._model = BGEM3FlagModel(
                self.config.model,
                device=self._device,
                use_fp16=(self._device == "cuda"),
            )
            logger.info(f"BGE-M3 loaded (dim={self._dimension}, sparse={self._use_sparse})")

        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model: {e}")

    def unload(self) -> None:
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
        return self._model is not None

    @property
    def vram_required(self) -> float:
        return VRAM_ESTIMATE

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def supports_sparse(self) -> bool:
        return self._use_sparse

    def _convert_sparse(self, sparse_dict: dict) -> SparseVector | None:
        """Convert FlagEmbedding sparse dict to SparseVector with int indices."""
        if not sparse_dict:
            return None
        # FlagEmbedding returns string keys - convert to int
        indices = [int(k) for k in sparse_dict.keys()]
        values = [float(v) for v in sparse_dict.values()]
        return SparseVector(indices=indices, values=values)

    @timed
    @require_loaded
    def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate embeddings for texts (dense + optional sparse)."""
        if not texts:
            return []

        try:
            output = self._model.encode(
                texts,
                batch_size=self.config.batch_size,
                return_dense=True,
                return_sparse=self._use_sparse,
                return_colbert_vecs=False,
            )

            results = []
            dense_embeddings = output["dense_vecs"]
            sparse_embeddings = output.get("lexical_weights") if self._use_sparse else None

            for i in range(len(texts)):
                dense = dense_embeddings[i].tolist()
                sparse = None
                if sparse_embeddings is not None:
                    sparse = self._convert_sparse(sparse_embeddings[i])
                results.append(EmbeddingResult(dense=dense, sparse=sparse))

            logger.debug(f"Generated {len(results)} embeddings (sparse={self._use_sparse})")
            return results

        except Exception as e:
            raise EmbeddingError(f"Embedding generation failed: {e}")

    @require_loaded
    def embed_query(self, query: str) -> EmbeddingResult:
        """Generate embedding for a single query."""
        try:
            output = self._model.encode(
                [query],
                batch_size=1,
                return_dense=True,
                return_sparse=self._use_sparse,
                return_colbert_vecs=False,
            )

            dense = output["dense_vecs"][0].tolist()
            sparse = None
            if self._use_sparse and "lexical_weights" in output:
                sparse = self._convert_sparse(output["lexical_weights"][0])

            return EmbeddingResult(dense=dense, sparse=sparse)

        except Exception as e:
            raise EmbeddingError(f"Query embedding failed: {e}")
