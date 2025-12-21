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
    """Main entry point for Audio RAG system.
    
    Provides unified interface for ingestion and querying.
    
    Usage:
        rag = AudioRAG.from_config(env="development")
        
        # Ingest audio
        result = rag.ingest("podcast.mp3")
        
        # Query
        response = rag.query("What did they say about AI?")
        
        # Query with audio response
        response = rag.query("Summarize the main points", generate_audio=True)
    """
    
    def __init__(self, config: AudioRAGConfig):
        self.config = config
        
        # Setup logging
        setup_logging(level=config.log_level)
        
        # Shared resource manager
        self.resource_manager = ResourceManager(config.resources)
        
        # Shared components (created once, used by both pipelines)
        self._embedder = None
        self._retriever = None
        
        # Pipelines (share resource manager and components)
        self._ingestion_pipeline = None
        self._query_pipeline = None
        
        # Ensure directories exist
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
        """Create AudioRAG instance from configuration files.
        
        Args:
            config_path: Optional specific config file
            env: Environment (development, production)
            config_dir: Directory containing config files
            
        Returns:
            Configured AudioRAG instance
        """
        config = load_config(
            config_path=config_path,
            env=env,
            config_dir=config_dir,
        )
        return cls(config)
    
    @property
    def embedder(self):
        """Shared embedder instance."""
        if self._embedder is None:
            self._embedder = EmbeddingsRegistry.create(
                self.config.embedding.backend,
                config=self.config.embedding,
            )
        return self._embedder
    
    @property
    def retriever(self):
        """Shared retriever instance."""
        if self._retriever is None:
            # Ensure embedder is loaded to get dimension
            if not self.embedder.is_loaded:
                self.embedder.load()
            
            self._retriever = RetrievalRegistry.create(
                self.config.retrieval.backend,
                config=self.config.retrieval,
                embedding_dim=self.embedder.dimension,
            )
        return self._retriever
    
    @property
    def ingestion_pipeline(self) -> IngestionPipeline:
        """Lazy-load ingestion pipeline with shared components."""
        if self._ingestion_pipeline is None:
            self._ingestion_pipeline = IngestionPipeline(
                config=self.config,
                resource_manager=self.resource_manager,
            )
            # Inject shared components
            self._ingestion_pipeline._embedder = self.embedder
            self._ingestion_pipeline._retriever = self.retriever
        return self._ingestion_pipeline
    
    @property
    def query_pipeline(self) -> QueryPipeline:
        """Lazy-load query pipeline with shared components."""
        if self._query_pipeline is None:
            self._query_pipeline = QueryPipeline(
                config=self.config,
                resource_manager=self.resource_manager,
            )
            # Inject shared components
            self._query_pipeline._embedder = self.embedder
            self._query_pipeline._retriever = self.retriever
        return self._query_pipeline
    
    def ingest(
        self,
        audio_path: Path | str,
        enable_diarization: bool = True,
        language: str | None = None,
    ) -> IngestionResult:
        """Ingest an audio file into the RAG system.
        
        Args:
            audio_path: Path to audio file (mp3, wav, etc.)
            enable_diarization: Whether to identify speakers
            language: Language code (None for auto-detect)
            
        Returns:
            IngestionResult with statistics
        """
        return self.ingestion_pipeline.ingest(
            audio_path=audio_path,
            enable_diarization=enable_diarization,
            language=language,
        )
    
    def ingest_batch(
        self,
        audio_paths: list[Path | str],
        enable_diarization: bool = True,
        language: str | None = None,
    ) -> list[IngestionResult]:
        """Ingest multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            enable_diarization: Whether to identify speakers
            language: Language code
            
        Returns:
            List of IngestionResults
        """
        results = []
        for i, path in enumerate(audio_paths, 1):
            logger.info(f"Processing file {i}/{len(audio_paths)}: {path}")
            try:
                result = self.ingest(
                    audio_path=path,
                    enable_diarization=enable_diarization,
                    language=language,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to ingest {path}: {e}")
                results.append(None)
        
        return results
    
    def query(
        self,
        query_text: str,
        top_k: int | None = None,
        generate_audio: bool = False,
        audio_output_path: Path | str | None = None,
    ) -> QueryResult:
        """Query the RAG system.
        
        Args:
            query_text: Natural language query
            top_k: Number of results to retrieve
            generate_audio: Whether to generate audio response
            audio_output_path: Path for audio output
            
        Returns:
            QueryResult with retrieved context and optional audio
        """
        return self.query_pipeline.query(
            query_text=query_text,
            top_k=top_k,
            generate_audio=generate_audio,
            audio_output_path=audio_output_path,
        )
    
    def get_context(self, query: str, top_k: int | None = None) -> str:
        """Get context for use with external LLM.
        
        Args:
            query: User query
            top_k: Number of results
            
        Returns:
            Formatted context string
        """
        return self.query_pipeline.get_context_for_llm(query, top_k=top_k)
    
    def status(self) -> dict:
        """Get system status."""
        return {
            "config": {
                "asr_backend": self.config.asr.backend,
                "asr_model": self.config.asr.model_size,
                "diarization_backend": self.config.diarization.backend,
                "embedding_backend": self.config.embedding.backend,
                "retrieval_backend": self.config.retrieval.backend,
                "tts_backend": self.config.tts.backend,
            },
            "resources": self.resource_manager.status(),
            "collection": {
                "name": self.config.retrieval.collection_name,
                "count": self.retriever.count() if self._retriever else 0,
            },
        }
    
    def clear_collection(self) -> None:
        """Clear all data from the vector store."""
        self.retriever.delete_collection()
        logger.info("Collection cleared")
    
    def unload_all(self) -> None:
        """Unload all models to free memory."""
        if self._ingestion_pipeline:
            # Only unload ASR and diarizer from ingestion
            if self._ingestion_pipeline._asr and self._ingestion_pipeline._asr.is_loaded:
                self._ingestion_pipeline._asr.unload()
            if self._ingestion_pipeline._diarizer and self._ingestion_pipeline._diarizer.is_loaded:
                self._ingestion_pipeline._diarizer.unload()
        
        if self._query_pipeline:
            # Only unload TTS from query
            if self._query_pipeline._tts and self._query_pipeline._tts.is_loaded:
                self._query_pipeline._tts.unload()
        
        # Unload shared embedder
        if self._embedder and self._embedder.is_loaded:
            self._embedder.unload()
        
        self.resource_manager.unload_all()
        logger.info("All models unloaded")
