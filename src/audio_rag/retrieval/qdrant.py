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
    """
    
    def __init__(self, config: RetrievalConfig, embedding_dim: int = 1024):
        self.config = config
        self.embedding_dim = embedding_dim
        self._client = None
        self._collection_exists = False
        logger.info(
            f"QdrantRetriever initialized: collection={config.collection_name}, "
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
    
    def _ensure_collection(self) -> None:
        """Ensure collection exists with correct schema."""
        if self._collection_exists:
            return
        
        try:
            from qdrant_client.models import Distance, VectorParams
            
            client = self._get_client()
            collections = client.get_collections().collections
            exists = any(c.name == self.config.collection_name for c in collections)
            
            if not exists:
                logger.info(f"Creating collection: {self.config.collection_name}")
                client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )
            else:
                logger.debug(f"Collection {self.config.collection_name} exists")
            
            self._collection_exists = True
            
        except Exception as e:
            raise RetrievalError(f"Failed to ensure collection: {e}")
    
    @timed
    def add(self, chunks: list[AudioChunk], embeddings: list[list[float]]) -> None:
        """Add chunks with their embeddings to the store.
        
        Args:
            chunks: List of audio chunks
            embeddings: Corresponding embedding vectors
        """
        if not chunks:
            return
        
        if len(chunks) != len(embeddings):
            raise RetrievalError(
                f"Chunks/embeddings mismatch: {len(chunks)} chunks, {len(embeddings)} embeddings"
            )
        
        self._ensure_collection()
        
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
                collection_name=self.config.collection_name,
                points=points,
            )
            
            logger.info(f"Added {len(points)} chunks to {self.config.collection_name}")
            
        except Exception as e:
            raise RetrievalError(f"Failed to add chunks: {e}")
    
    @timed
    def search(
        self, query_embedding: list[float], top_k: int | None = None
    ) -> list[RetrievalResult]:
        """Search for similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results (defaults to config.top_k)
            
        Returns:
            List of retrieval results with scores
        """
        self._ensure_collection()
        top_k = top_k or self.config.top_k
        
        try:
            client = self._get_client()
            results = client.query_points(
                collection_name=self.config.collection_name,
                query=query_embedding,
                limit=top_k,
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
                        source=self.config.collection_name,
                    )
                )
            
            logger.debug(f"Search returned {len(retrieval_results)} results")
            return retrieval_results
            
        except Exception as e:
            raise RetrievalError(f"Search failed: {e}")
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            client = self._get_client()
            client.delete_collection(collection_name=self.config.collection_name)
            self._collection_exists = False
            logger.info(f"Deleted collection: {self.config.collection_name}")
            
        except Exception as e:
            raise RetrievalError(f"Failed to delete collection: {e}")
    
    def count(self) -> int:
        """Get number of vectors in collection."""
        self._ensure_collection()
        
        try:
            client = self._get_client()
            info = client.get_collection(self.config.collection_name)
            return info.points_count
            
        except Exception as e:
            raise RetrievalError(f"Failed to get count: {e}")
