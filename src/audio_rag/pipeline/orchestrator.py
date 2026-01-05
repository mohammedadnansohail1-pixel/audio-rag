"""Main orchestrator for Audio RAG system."""

from pathlib import Path

from audio_rag.pipeline.ingestion import IngestionPipeline, IngestionResult
from audio_rag.pipeline.query import QueryPipeline, QueryResult
from audio_rag.embeddings import EmbeddingsRegistry
from audio_rag.retrieval import RetrievalRegistry
from audio_rag.resources import ResourceManager
from audio_rag.config import AudioRAGConfig, load_config
from audio_rag.utils import get_logger, setup_logging

logger = get_logger(__name__)


class AudioRAG:
    """Main entry point for Audio RAG system."""

    def __init__(self, config: AudioRAGConfig):
        self.config = config
        setup_logging(level=config.log_level)
        self.resource_manager = ResourceManager(config.resources)
        self._embedder = None
        self._retriever = None
        self._ingestion_pipeline = None
        self._query_pipeline = None
        Path(config.data_dir).mkdir(parents=True, exist_ok=True)
        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
        logger.info("AudioRAG initialized")

    @classmethod
    def from_config(
        cls,
        config_path: Path | str | None = None,
        env: str | None = None,
        config_dir: Path | str = "configs",
    ) -> "AudioRAG":
        config = load_config(config_path=config_path, env=env, config_dir=config_dir)
        return cls(config)

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
    def ingestion_pipeline(self) -> IngestionPipeline:
        if self._ingestion_pipeline is None:
            self._ingestion_pipeline = IngestionPipeline(
                config=self.config, resource_manager=self.resource_manager)
            self._ingestion_pipeline._embedder = self.embedder
            self._ingestion_pipeline._retriever = self.retriever
        return self._ingestion_pipeline

    @property
    def query_pipeline(self) -> QueryPipeline:
        if self._query_pipeline is None:
            self._query_pipeline = QueryPipeline(
                config=self.config, resource_manager=self.resource_manager)
            self._query_pipeline._embedder = self.embedder
            self._query_pipeline._retriever = self.retriever
        return self._query_pipeline

    def ingest(
        self,
        audio_path: Path | str,
        collection_name: str | None = None,
        metadata: dict | None = None,
        enable_diarization: bool = True,
        language: str | None = None,
    ) -> IngestionResult:
        return self.ingestion_pipeline.ingest(
            audio_path=audio_path,
            collection_name=collection_name,
            metadata=metadata,
            enable_diarization=enable_diarization,
            language=language)

    def ingest_batch(
        self,
        audio_paths: list[Path | str],
        collection_name: str | None = None,
        metadata: dict | None = None,
        enable_diarization: bool = True,
        language: str | None = None,
    ) -> list[IngestionResult]:
        results = []
        for i, path in enumerate(audio_paths, 1):
            logger.info(f"Processing file {i}/{len(audio_paths)}: {path}")
            try:
                result = self.ingest(
                    audio_path=path, collection_name=collection_name,
                    metadata=metadata, enable_diarization=enable_diarization,
                    language=language)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to ingest {path}: {e}")
                results.append(None)
        return results

    def query(
        self,
        query_text: str,
        collection_name: str | None = None,
        top_k: int | None = None,
        filter_metadata: dict | None = None,
        search_type: str | None = None,
        enable_reranking: bool = True,
        generate_answer: bool = True,
        generate_audio: bool = False,
        audio_output_path: Path | str | None = None,
    ) -> QueryResult:
        return self.query_pipeline.query(
            query_text=query_text,
            collection_name=collection_name,
            top_k=top_k,
            filter_metadata=filter_metadata,
            search_type=search_type,
            enable_reranking=enable_reranking,
            generate_answer=generate_answer,
            generate_audio=generate_audio,
            audio_output_path=audio_output_path)

    def get_context(
        self,
        query: str,
        collection_name: str | None = None,
        top_k: int | None = None,
        filter_metadata: dict | None = None,
    ) -> str:
        return self.query_pipeline.get_context_for_llm(
            query=query, collection_name=collection_name,
            top_k=top_k, filter_metadata=filter_metadata)

    def status(self) -> dict:
        return {
            "config": {
                "asr_backend": self.config.asr.backend,
                "asr_model": self.config.asr.model_size,
                "diarization_backend": self.config.diarization.backend,
                "embedding_backend": self.config.embedding.backend,
                "embedding_sparse": self.config.embedding.use_sparse,
                "retrieval_backend": self.config.retrieval.backend,
                "retrieval_search_type": self.config.retrieval.search_type,
                "reranking_backend": self.config.reranking.backend,
                "generation_backend": self.config.generation.backend,
                "generation_model": self.config.generation.model,
                "tts_backend": self.config.tts.backend,
            },
            "resources": self.resource_manager.status(),
            "collection": {
                "name": self.config.retrieval.collection_name,
                "count": self.retriever.count() if self._retriever else 0,
            },
        }

    def clear_collection(self, collection_name: str | None = None) -> None:
        self.retriever.delete_collection(collection_name)
        logger.info(f"Collection cleared: {collection_name or self.config.retrieval.collection_name}")

    def unload_all(self) -> None:
        if self._ingestion_pipeline:
            if self._ingestion_pipeline._asr and self._ingestion_pipeline._asr.is_loaded:
                self._ingestion_pipeline._asr.unload()
            if self._ingestion_pipeline._diarizer and self._ingestion_pipeline._diarizer.is_loaded:
                self._ingestion_pipeline._diarizer.unload()
        if self._query_pipeline:
            if self._query_pipeline._tts and self._query_pipeline._tts.is_loaded:
                self._query_pipeline._tts.unload()
        if self._embedder and self._embedder.is_loaded:
            self._embedder.unload()
        self.resource_manager.unload_all()
        logger.info("All models unloaded")
