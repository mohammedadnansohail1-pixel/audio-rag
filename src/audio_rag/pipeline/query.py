"""Query pipeline - Query → Retrieve → Generate → Response."""

from pathlib import Path
from dataclasses import dataclass

from audio_rag.core import RetrievalResult, PipelineError
from audio_rag.embeddings import EmbeddingsRegistry
from audio_rag.retrieval import RetrievalRegistry
from audio_rag.generation import GeneratorRegistry
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
    generated_answer: str | None = None
    audio_path: Path | None = None


class QueryPipeline:
    """Pipeline for querying the audio RAG system."""

    def __init__(self, config: AudioRAGConfig, resource_manager: ResourceManager | None = None):
        self.config = config
        self.resource_manager = resource_manager or ResourceManager(config.resources)
        self._embedder = None
        self._retriever = None
        self._generator = None
        self._tts = None
        logger.info("QueryPipeline initialized")

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = EmbeddingsRegistry.create(
                self.config.embedding.backend, config=self.config.embedding)
        return self._embedder

    @property
    def retriever(self):
        if self._retriever is None:
            if not self.embedder.is_loaded:
                self.embedder.load()
            self._retriever = RetrievalRegistry.create(
                self.config.retrieval.backend,
                config=self.config.retrieval,
                embedding_dim=self.embedder.dimension)
        return self._retriever

    @property
    def generator(self):
        if self._generator is None:
            self._generator = GeneratorRegistry.create(
                self.config.generation.backend, config=self.config.generation)
        return self._generator

    @property
    def tts(self):
        if self._tts is None:
            self._tts = TTSRegistry.create(
                self.config.tts.backend, config=self.config.tts)
        return self._tts

    @timed
    def query(
        self,
        query_text: str,
        collection_name: str | None = None,
        top_k: int | None = None,
        filter_metadata: dict | None = None,
        generate_answer: bool = True,
        generate_audio: bool = False,
        audio_output_path: Path | str | None = None,
    ) -> QueryResult:
        """Query the audio RAG system."""
        resolved_collection = collection_name or self.config.retrieval.collection_name
        logger.info(f"Query: '{query_text[:50]}...' → {resolved_collection}")

        try:
            # Embed query
            self.resource_manager.ensure_vram(self.embedder.vram_required)
            query_embedding = self.embedder.embed_query(query_text)

            # Retrieve
            results = self.retriever.search(
                query_embedding, top_k=top_k,
                collection_name=resolved_collection,
                filter_metadata=filter_metadata)
            logger.info(f"Retrieved {len(results)} results")

            if not results:
                return QueryResult(query=query_text, collection_name=resolved_collection, results=[])

            # Build raw response
            response_text = self._build_response(query_text, results)

            # LLM generation
            generated_answer = None
            if generate_answer and self.generator is not None:
                try:
                    generated_answer = self.generator.generate(query_text, results)
                    logger.info(f"Generated answer: {len(generated_answer)} chars")
                except Exception as e:
                    logger.warning(f"Generation failed: {e}")

            # TTS
            audio_path = None
            if generate_audio:
                tts_text = generated_answer or response_text
                if tts_text:
                    audio_output_path = Path(audio_output_path or "./output/response.wav")
                    self.resource_manager.ensure_vram(self.tts.vram_required)
                    audio_path = self.tts.synthesize(tts_text, audio_output_path)

            return QueryResult(
                query=query_text,
                collection_name=resolved_collection,
                results=results,
                response_text=response_text,
                generated_answer=generated_answer,
                audio_path=audio_path)

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise PipelineError(f"Query failed: {e}") from e

    def _build_response(self, query: str, results: list[RetrievalResult]) -> str:
        if not results:
            return "No relevant information found."
        parts = []
        for result in results:
            chunk = result.chunk
            speaker = chunk.speaker or "Unknown"
            time_str = f"{chunk.start:.1f}s-{chunk.end:.1f}s"
            parts.append(f"[{speaker} at {time_str}]: {chunk.text}")
        return "\n\n".join(parts)

    def get_context_for_llm(
        self, query: str, collection_name: str | None = None,
        top_k: int | None = None, filter_metadata: dict | None = None,
    ) -> str:
        resolved_collection = collection_name or self.config.retrieval.collection_name
        try:
            self.resource_manager.ensure_vram(self.embedder.vram_required)
            query_embedding = self.embedder.embed_query(query)
            results = self.retriever.search(
                query_embedding, top_k=top_k,
                collection_name=resolved_collection,
                filter_metadata=filter_metadata)

            if not results:
                return "No relevant context found."

            parts = []
            for result in results:
                chunk = result.chunk
                speaker = chunk.speaker or "Speaker"
                source = chunk.metadata.get("source_filename", "unknown") if chunk.metadata else "unknown"
                parts.append(
                    f"<context speaker=\"{speaker}\" start=\"{chunk.start:.1f}\" "
                    f"end=\"{chunk.end:.1f}\" source=\"{source}\" score=\"{result.score:.3f}\">\n"
                    f"{chunk.text}\n</context>")
            return "\n\n".join(parts)
        except Exception as e:
            raise PipelineError(f"Failed to get context: {e}") from e

    def unload_all(self) -> None:
        if self._embedder and self._embedder.is_loaded:
            self._embedder.unload()
        if self._tts and self._tts.is_loaded:
            self._tts.unload()
        logger.info("Query pipeline models unloaded")
