"""Query pipeline - Query → Retrieve → Response."""

from pathlib import Path
from dataclasses import dataclass

from audio_rag.core import RetrievalResult, PipelineError
from audio_rag.embeddings import EmbeddingsRegistry
from audio_rag.retrieval import RetrievalRegistry
from audio_rag.tts import TTSRegistry
from audio_rag.resources import ResourceManager
from audio_rag.config import AudioRAGConfig
from audio_rag.utils import get_logger, timed

logger = get_logger(__name__)


@dataclass
class QueryResult:
    """Result of a query."""
    query: str
    collection_name: str
    results: list[RetrievalResult]
    response_text: str | None = None
    audio_path: Path | None = None


class QueryPipeline:
    """Pipeline for querying the audio RAG system.

    Flow: Query → Embed → Retrieve → (Optional: Generate Response) → (Optional: TTS)
    
    Supports multi-tenancy via collection_name parameter.
    """

    def __init__(self, config: AudioRAGConfig, resource_manager: ResourceManager | None = None):
        self.config = config
        self.resource_manager = resource_manager or ResourceManager(config.resources)

        # Lazy-loaded components
        self._embedder = None
        self._retriever = None
        self._tts = None

        logger.info("QueryPipeline initialized")

    @property
    def embedder(self):
        """Lazy-load embedder."""
        if self._embedder is None:
            self._embedder = EmbeddingsRegistry.create(
                self.config.embedding.backend,
                config=self.config.embedding,
            )
        return self._embedder

    @property
    def retriever(self):
        """Lazy-load retriever."""
        if self._retriever is None:
            if not self.embedder.is_loaded:
                self.embedder.load()

            self._retriever = RetrievalRegistry.create(
                self.config.retrieval.backend,
                config=self.config.retrieval,
                embedding_dim=self.embedder.dimension,
            )
        return self._retriever

    @property
    def tts(self):
        """Lazy-load TTS."""
        if self._tts is None:
            self._tts = TTSRegistry.create(
                self.config.tts.backend,
                config=self.config.tts,
            )
        return self._tts

    @timed
    def query(
        self,
        query_text: str,
        collection_name: str | None = None,
        top_k: int | None = None,
        filter_metadata: dict | None = None,
        generate_audio: bool = False,
        audio_output_path: Path | str | None = None,
    ) -> QueryResult:
        """Query the audio RAG system.

        Args:
            query_text: Natural language query
            collection_name: Target collection (tenant isolation). Uses config default if None.
            top_k: Number of results to retrieve
            filter_metadata: Optional metadata filter for results
            generate_audio: Whether to generate audio response
            audio_output_path: Path for audio output (if generate_audio)

        Returns:
            QueryResult with retrieved chunks and optional audio
            
        Raises:
            PipelineError: If query fails
        """
        # Resolve collection name
        resolved_collection = collection_name or self.config.retrieval.collection_name
        
        logger.info(f"Query: '{query_text[:50]}...' → collection={resolved_collection}")

        try:
            # Step 1: Embed query
            self.resource_manager.ensure_vram(self.embedder.vram_required)
            query_embedding = self.embedder.embed_query(query_text)

            # Step 2: Retrieve with collection override
            results = self.retriever.search(
                query_embedding,
                top_k=top_k,
                collection_name=resolved_collection,
                filter_metadata=filter_metadata,
            )
            logger.info(f"Retrieved {len(results)} results from {resolved_collection}")

            if not results:
                return QueryResult(
                    query=query_text,
                    collection_name=resolved_collection,
                    results=[],
                )

            # Step 3: Build response text from retrieved chunks
            response_text = self._build_response(query_text, results)

            # Step 4: Optional TTS
            audio_path = None
            if generate_audio and response_text:
                if audio_output_path is None:
                    audio_output_path = Path("./output/response.wav")
                audio_output_path = Path(audio_output_path)

                self.resource_manager.ensure_vram(self.tts.vram_required)
                audio_path = self.tts.synthesize(response_text, audio_output_path)
                logger.info(f"Generated audio: {audio_path}")

            return QueryResult(
                query=query_text,
                collection_name=resolved_collection,
                results=results,
                response_text=response_text,
                audio_path=audio_path,
            )
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise PipelineError(f"Query failed: {e}") from e

    def _build_response(self, query: str, results: list[RetrievalResult]) -> str:
        """Build response text from retrieved results.

        This is a simple concatenation. In production, you'd use an LLM
        to synthesize a coherent response.
        """
        if not results:
            return "No relevant information found."

        # Group by speaker if available
        response_parts = []
        for i, result in enumerate(results, 1):
            chunk = result.chunk
            speaker = chunk.speaker or "Unknown"
            score = result.score

            # Format: include speaker and timestamp
            time_str = f"{chunk.start:.1f}s-{chunk.end:.1f}s"
            response_parts.append(
                f"[{speaker} at {time_str}]: {chunk.text}"
            )

        return "\n\n".join(response_parts)

    def get_context_for_llm(
        self,
        query: str,
        collection_name: str | None = None,
        top_k: int | None = None,
        filter_metadata: dict | None = None,
    ) -> str:
        """Get retrieved context formatted for LLM input.

        Args:
            query: User query
            collection_name: Target collection (tenant isolation)
            top_k: Number of results
            filter_metadata: Optional metadata filter

        Returns:
            Formatted context string for LLM
        """
        resolved_collection = collection_name or self.config.retrieval.collection_name
        
        try:
            self.resource_manager.ensure_vram(self.embedder.vram_required)
            query_embedding = self.embedder.embed_query(query)
            results = self.retriever.search(
                query_embedding,
                top_k=top_k,
                collection_name=resolved_collection,
                filter_metadata=filter_metadata,
            )

            if not results:
                return "No relevant context found."

            context_parts = []
            for result in results:
                chunk = result.chunk
                speaker = chunk.speaker or "Speaker"
                source = chunk.metadata.get("source_filename", "unknown") if chunk.metadata else "unknown"
                context_parts.append(
                    f"<context speaker=\"{speaker}\" "
                    f"start=\"{chunk.start:.1f}\" end=\"{chunk.end:.1f}\" "
                    f"source=\"{source}\" score=\"{result.score:.3f}\">\n"
                    f"{chunk.text}\n</context>"
                )

            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Failed to get context: {e}")
            raise PipelineError(f"Failed to get context: {e}") from e

    def unload_all(self) -> None:
        """Unload all models to free memory."""
        if self._embedder and self._embedder.is_loaded:
            self._embedder.unload()
        if self._tts and self._tts.is_loaded:
            self._tts.unload()

        logger.info("Query pipeline models unloaded")
