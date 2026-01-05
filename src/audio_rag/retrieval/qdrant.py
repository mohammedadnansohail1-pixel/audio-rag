"""Qdrant vector store with hybrid search support."""

from uuid import uuid4

from audio_rag.retrieval.base import RetrievalRegistry
from audio_rag.core import BaseRetriever, AudioChunk, RetrievalResult, RetrievalError
from audio_rag.core import EmbeddingResult, SparseVector
from audio_rag.config import RetrievalConfig
from audio_rag.utils import get_logger, timed

logger = get_logger(__name__)


@RetrievalRegistry.register("qdrant")
class QdrantRetriever(BaseRetriever):
    """Qdrant vector store with dense, sparse, and hybrid search.

    Supports:
    - Dense search (cosine similarity)
    - Sparse search (BM25-like lexical)
    - Hybrid search (RRF fusion of dense + sparse)
    """

    def __init__(self, config: RetrievalConfig, embedding_dim: int = 1024):
        self.config = config
        self.embedding_dim = embedding_dim
        self._client = None
        self._existing_collections: set[str] = set()
        self._hybrid_collections: set[str] = set()  # Track hybrid-enabled collections
        logger.info(
            f"QdrantRetriever initialized: collection={config.collection_name}, "
            f"search_type={config.search_type}"
        )

    def _get_client(self):
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
        return collection_name or self.config.collection_name

    def _ensure_collection(self, collection_name: str | None = None, hybrid: bool = False) -> str:
        """Ensure collection exists with correct schema.

        Args:
            collection_name: Optional override
            hybrid: Whether to create hybrid (dense + sparse) collection

        Returns:
            Resolved collection name
        """
        resolved = self._resolve_collection(collection_name)
        
        # Check if already verified
        if resolved in self._existing_collections:
            if hybrid and resolved not in self._hybrid_collections:
                logger.warning(
                    f"Collection {resolved} exists but is not hybrid-enabled. "
                    "Re-index required for hybrid search."
                )
            return resolved

        try:
            from qdrant_client.models import (
                Distance, VectorParams, 
                SparseVectorParams, SparseIndexParams
            )

            client = self._get_client()
            collections = client.get_collections().collections
            exists = any(c.name == resolved for c in collections)

            if not exists:
                logger.info(f"Creating {'hybrid' if hybrid else 'dense'} collection: {resolved}")
                
                if hybrid:
                    # Hybrid collection with named vectors
                    client.create_collection(
                        collection_name=resolved,
                        vectors_config={
                            "dense": VectorParams(
                                size=self.embedding_dim,
                                distance=Distance.COSINE,
                            ),
                        },
                        sparse_vectors_config={
                            "sparse": SparseVectorParams(
                                index=SparseIndexParams(on_disk=False),
                            ),
                        },
                    )
                    self._hybrid_collections.add(resolved)
                else:
                    # Dense-only collection (backward compatible)
                    client.create_collection(
                        collection_name=resolved,
                        vectors_config=VectorParams(
                            size=self.embedding_dim,
                            distance=Distance.COSINE,
                        ),
                    )
            else:
                # Check if existing collection is hybrid
                info = client.get_collection(resolved)
                if info.config.params.sparse_vectors:
                    self._hybrid_collections.add(resolved)
                    logger.debug(f"Collection {resolved} is hybrid-enabled")
                else:
                    logger.debug(f"Collection {resolved} is dense-only")

            self._existing_collections.add(resolved)
            return resolved

        except Exception as e:
            raise RetrievalError(f"Failed to ensure collection '{resolved}': {e}")

    def is_hybrid_collection(self, collection_name: str | None = None) -> bool:
        """Check if collection supports hybrid search."""
        resolved = self._resolve_collection(collection_name)
        self._ensure_collection(resolved)
        return resolved in self._hybrid_collections

    @timed
    def add(
        self,
        chunks: list[AudioChunk],
        embeddings: list[EmbeddingResult],
        collection_name: str | None = None,
    ) -> None:
        """Add chunks with embeddings to the store.

        Args:
            chunks: List of audio chunks
            embeddings: EmbeddingResults with dense (and optional sparse) vectors
            collection_name: Optional collection override
        """
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise RetrievalError(
                f"Chunks/embeddings mismatch: {len(chunks)} chunks, {len(embeddings)} embeddings"
            )

        # Check if we have sparse vectors
        has_sparse = any(e.sparse is not None for e in embeddings)
        resolved = self._ensure_collection(collection_name, hybrid=has_sparse)

        try:
            from qdrant_client.models import PointStruct, SparseVector as QdrantSparseVector

            client = self._get_client()
            is_hybrid = resolved in self._hybrid_collections

            points = []
            for chunk, emb in zip(chunks, embeddings):
                point_id = str(uuid4())
                payload = {
                    "text": chunk.text,
                    "start": chunk.start,
                    "end": chunk.end,
                    "speaker": chunk.speaker,
                    "metadata": chunk.metadata or {},
                }

                if is_hybrid:
                    # Named vectors for hybrid collection
                    vector = {"dense": emb.dense}
                    if emb.sparse is not None:
                        sparse_vec = QdrantSparseVector(
                            indices=emb.sparse.indices,
                            values=emb.sparse.values,
                        )
                        points.append(PointStruct(
                            id=point_id,
                            vector=vector,
                            payload=payload,
                        ))
                        # Update sparse separately
                        client.upsert(
                            collection_name=resolved,
                            points=[PointStruct(
                                id=point_id,
                                vector={
                                    "dense": emb.dense,
                                    "sparse": sparse_vec,
                                },
                                payload=payload,
                            )],
                        )
                        continue
                    else:
                        points.append(PointStruct(
                            id=point_id, vector=vector, payload=payload
                        ))
                else:
                    # Legacy dense-only
                    points.append(PointStruct(
                        id=point_id, vector=emb.dense, payload=payload
                    ))

            if points:
                client.upsert(collection_name=resolved, points=points)

            logger.info(f"Added {len(chunks)} chunks to {resolved} (hybrid={is_hybrid})")

        except Exception as e:
            raise RetrievalError(f"Failed to add chunks to '{resolved}': {e}")

    @timed
    def search(
        self,
        query_embedding: EmbeddingResult,
        top_k: int | None = None,
        collection_name: str | None = None,
        filter_metadata: dict | None = None,
        search_type: str | None = None,
    ) -> list[RetrievalResult]:
        """Search for similar chunks.

        Args:
            query_embedding: EmbeddingResult with dense (and optional sparse)
            top_k: Number of results
            collection_name: Optional collection override
            filter_metadata: Optional metadata filter
            search_type: Override search type (dense/sparse/hybrid)

        Returns:
            List of retrieval results with scores
        """
        resolved = self._ensure_collection(collection_name)
        top_k = top_k or self.config.top_k
        search_type = search_type or self.config.search_type

        try:
            from qdrant_client.models import (
                Filter, FieldCondition, MatchValue,
                SparseVector as QdrantSparseVector,
                Prefetch, FusionQuery, Fusion
            )

            client = self._get_client()
            is_hybrid = resolved in self._hybrid_collections

            # Build filter
            query_filter = None
            if filter_metadata:
                conditions = [
                    FieldCondition(key=f"metadata.{k}", match=MatchValue(value=v))
                    for k, v in filter_metadata.items()
                ]
                query_filter = Filter(must=conditions)

            # Determine search strategy
            if search_type == "hybrid" and is_hybrid and query_embedding.sparse:
                # Hybrid search with RRF fusion
                logger.debug("Using hybrid search (dense + sparse)")
                
                sparse_vec = QdrantSparseVector(
                    indices=query_embedding.sparse.indices,
                    values=query_embedding.sparse.values,
                )
                
                results = client.query_points(
                    collection_name=resolved,
                    prefetch=[
                        Prefetch(
                            query=query_embedding.dense,
                            using="dense",
                            limit=top_k * 2,
                        ),
                        Prefetch(
                            query=sparse_vec,
                            using="sparse",
                            limit=top_k * 2,
                        ),
                    ],
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=top_k,
                    query_filter=query_filter,
                )
            elif search_type == "sparse" and is_hybrid and query_embedding.sparse:
                # Sparse-only search
                logger.debug("Using sparse-only search")
                sparse_vec = QdrantSparseVector(
                    indices=query_embedding.sparse.indices,
                    values=query_embedding.sparse.values,
                )
                results = client.query_points(
                    collection_name=resolved,
                    query=sparse_vec,
                    using="sparse",
                    limit=top_k,
                    query_filter=query_filter,
                )
            else:
                # Dense search (default or fallback)
                logger.debug("Using dense search")
                if is_hybrid:
                    results = client.query_points(
                        collection_name=resolved,
                        query=query_embedding.dense,
                        using="dense",
                        limit=top_k,
                        query_filter=query_filter,
                    )
                else:
                    # Legacy dense-only collection
                    results = client.query_points(
                        collection_name=resolved,
                        query=query_embedding.dense,
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
                retrieval_results.append(RetrievalResult(
                    chunk=chunk, score=hit.score, source=resolved
                ))

            logger.debug(f"Search returned {len(retrieval_results)} results")
            return retrieval_results

        except Exception as e:
            raise RetrievalError(f"Search failed in '{resolved}': {e}")

    def delete_collection(self, collection_name: str | None = None) -> None:
        resolved = self._resolve_collection(collection_name)
        try:
            client = self._get_client()
            client.delete_collection(collection_name=resolved)
            self._existing_collections.discard(resolved)
            self._hybrid_collections.discard(resolved)
            logger.info(f"Deleted collection: {resolved}")
        except Exception as e:
            raise RetrievalError(f"Failed to delete collection '{resolved}': {e}")

    def count(self, collection_name: str | None = None) -> int:
        resolved = self._ensure_collection(collection_name)
        try:
            client = self._get_client()
            info = client.get_collection(resolved)
            return info.points_count
        except Exception as e:
            raise RetrievalError(f"Failed to get count for '{resolved}': {e}")

    def collection_exists(self, collection_name: str | None = None) -> bool:
        resolved = self._resolve_collection(collection_name)
        try:
            client = self._get_client()
            collections = client.get_collections().collections
            return any(c.name == resolved for c in collections)
        except Exception as e:
            raise RetrievalError(f"Failed to check collection '{resolved}': {e}")
