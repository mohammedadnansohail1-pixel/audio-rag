"""Qdrant vector store retrieval implementation."""

from uuid import uuid4

from audio_rag.retrieval.base import RetrievalRegistry
from audio_rag.core import BaseRetriever, AudioChunk, RetrievalResult, RetrievalError
from audio_rag.config import RetrievalConfig
from audio_rag.utils import get_logger, timed

logger = get_logger(__name__)


@RetrievalRegistry.register("qdrant")
class QdrantRetriever(BaseRetriever):
    """Qdrant vector store retrieval backend.

    Supports dense, sparse, and hybrid search modes.
    Multi-tenant via collection_name override at runtime.
    """

    def __init__(self, config: RetrievalConfig, embedding_dim: int = 1024):
        self.config = config
        self.embedding_dim = embedding_dim
        self._client = None
        self._existing_collections: set[str] = set()
        logger.info(
            f"QdrantRetriever initialized: default_collection={config.collection_name}, "
            f"search_type={config.search_type}"
        )

    def _get_client(self):
        """Get or create Qdrant client."""
        if self._client is not None:
            return self._client

        try:
            from qdrant_client import QdrantClient

            if self.config.qdrant_in_memory:
                logger.info("Using in-memory Qdrant")
                self._client = QdrantClient(":memory:")
            else:
                logger.info(f"Connecting to Qdrant at {self.config.qdrant_host}:{self.config.qdrant_port}")
                self._client = QdrantClient(
                    host=self.config.qdrant_host,
                    port=self.config.qdrant_port,
                )

            return self._client

        except Exception as e:
            raise RetrievalError(f"Failed to connect to Qdrant: {e}")

    def _resolve_collection(self, collection_name: str | None) -> str:
        """Resolve collection name - use override or default from config."""
        return collection_name or self.config.collection_name

    def _ensure_collection(self, collection_name: str | None = None) -> str:
        """Ensure collection exists with correct schema.
        
        Args:
            collection_name: Optional override (uses config default if None)
            
        Returns:
            Resolved collection name
        """
        resolved = self._resolve_collection(collection_name)
        
        if resolved in self._existing_collections:
            return resolved

        try:
            from qdrant_client.models import Distance, VectorParams

            client = self._get_client()
            collections = client.get_collections().collections
            exists = any(c.name == resolved for c in collections)

            if not exists:
                logger.info(f"Creating collection: {resolved}")
                client.create_collection(
                    collection_name=resolved,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )
            else:
                logger.debug(f"Collection {resolved} exists")

            self._existing_collections.add(resolved)
            return resolved

        except Exception as e:
            raise RetrievalError(f"Failed to ensure collection '{resolved}': {e}")

    @timed
    def add(
        self,
        chunks: list[AudioChunk],
        embeddings: list[list[float]],
        collection_name: str | None = None,
    ) -> None:
        """Add chunks with their embeddings to the store.

        Args:
            chunks: List of audio chunks
            embeddings: Corresponding embedding vectors
            collection_name: Optional collection override (for multi-tenancy)
        """
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise RetrievalError(
                f"Chunks/embeddings mismatch: {len(chunks)} chunks, {len(embeddings)} embeddings"
            )

        resolved = self._ensure_collection(collection_name)

        try:
            from qdrant_client.models import PointStruct

            points = []
            for chunk, embedding in zip(chunks, embeddings):
                point_id = str(uuid4())
                payload = {
                    "text": chunk.text,
                    "start": chunk.start,
                    "end": chunk.end,
                    "speaker": chunk.speaker,
                    "metadata": chunk.metadata or {},
                }
                points.append(
                    PointStruct(id=point_id, vector=embedding, payload=payload)
                )

            client = self._get_client()
            client.upsert(
                collection_name=resolved,
                points=points,
            )

            logger.info(f"Added {len(points)} chunks to {resolved}")

        except Exception as e:
            raise RetrievalError(f"Failed to add chunks to '{resolved}': {e}")

    @timed
    def search(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        collection_name: str | None = None,
        filter_metadata: dict | None = None,
    ) -> list[RetrievalResult]:
        """Search for similar chunks.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results (defaults to config.top_k)
            collection_name: Optional collection override (for multi-tenancy)
            filter_metadata: Optional metadata filter

        Returns:
            List of retrieval results with scores
        """
        resolved = self._ensure_collection(collection_name)
        top_k = top_k or self.config.top_k

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            client = self._get_client()
            
            # Build filter if metadata provided
            query_filter = None
            if filter_metadata:
                conditions = [
                    FieldCondition(
                        key=f"metadata.{key}",
                        match=MatchValue(value=value),
                    )
                    for key, value in filter_metadata.items()
                ]
                query_filter = Filter(must=conditions)

            results = client.query_points(
                collection_name=resolved,
                query=query_embedding,
                limit=top_k,
                query_filter=query_filter,
                score_threshold=self.config.score_threshold if self.config.score_threshold > 0 else None,
            )

            retrieval_results = []
            for hit in results.points:
                payload = hit.payload
                chunk = AudioChunk(
                    text=payload.get("text", ""),
                    start=payload.get("start", 0.0),
                    end=payload.get("end", 0.0),
                    speaker=payload.get("speaker"),
                    metadata=payload.get("metadata"),
                )
                retrieval_results.append(
                    RetrievalResult(
                        chunk=chunk,
                        score=hit.score,
                        source=resolved,
                    )
                )

            logger.debug(f"Search in '{resolved}' returned {len(retrieval_results)} results")
            return retrieval_results

        except Exception as e:
            raise RetrievalError(f"Search failed in '{resolved}': {e}")

    def delete_collection(self, collection_name: str | None = None) -> None:
        """Delete a collection.
        
        Args:
            collection_name: Optional override (uses config default if None)
        """
        resolved = self._resolve_collection(collection_name)
        
        try:
            client = self._get_client()
            client.delete_collection(collection_name=resolved)
            self._existing_collections.discard(resolved)
            logger.info(f"Deleted collection: {resolved}")

        except Exception as e:
            raise RetrievalError(f"Failed to delete collection '{resolved}': {e}")

    def count(self, collection_name: str | None = None) -> int:
        """Get number of vectors in collection.
        
        Args:
            collection_name: Optional override (uses config default if None)
        """
        resolved = self._ensure_collection(collection_name)

        try:
            client = self._get_client()
            info = client.get_collection(resolved)
            return info.points_count

        except Exception as e:
            raise RetrievalError(f"Failed to get count for '{resolved}': {e}")

    def collection_exists(self, collection_name: str | None = None) -> bool:
        """Check if collection exists.
        
        Args:
            collection_name: Optional override (uses config default if None)
        """
        resolved = self._resolve_collection(collection_name)
        
        try:
            client = self._get_client()
            collections = client.get_collections().collections
            return any(c.name == resolved for c in collections)
        except Exception as e:
            raise RetrievalError(f"Failed to check collection '{resolved}': {e}")
