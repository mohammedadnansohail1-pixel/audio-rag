"""Audio ingestion pipeline - Audio → Chunks → (Context) → Vectors."""

from pathlib import Path
from dataclasses import dataclass, field

from audio_rag.core import TranscriptSegment, AudioChunk, PipelineError
from audio_rag.asr import ASRRegistry
from audio_rag.diarization import DiarizationRegistry
from audio_rag.alignment import align_words_to_speakers, build_speaker_transcript
from audio_rag.chunking import ChunkingRegistry
from audio_rag.contextual import ContextualProcessor
from audio_rag.embeddings import EmbeddingsRegistry
from audio_rag.retrieval import RetrievalRegistry
from audio_rag.resources import ResourceManager
from audio_rag.config import AudioRAGConfig
from audio_rag.utils import get_logger, timed

logger = get_logger(__name__)


@dataclass
class IngestionResult:
    """Result of audio ingestion."""
    audio_path: str
    collection_name: str
    num_segments: int
    num_chunks: int
    duration_seconds: float
    speakers: list[str]
    language: str | None
    contextualized: bool = False
    metadata: dict = field(default_factory=dict)


class IngestionPipeline:
    """Pipeline for ingesting audio into the vector store."""

    def __init__(self, config: AudioRAGConfig, resource_manager: ResourceManager | None = None):
        self.config = config
        self.resource_manager = resource_manager or ResourceManager(config.resources)
        self._asr = None
        self._diarizer = None
        self._chunker = None
        self._contextual = None
        self._embedder = None
        self._retriever = None
        logger.info("IngestionPipeline initialized")

    @property
    def asr(self):
        if self._asr is None:
            self._asr = ASRRegistry.create(self.config.asr.backend, config=self.config.asr)
        return self._asr

    @property
    def diarizer(self):
        if self._diarizer is None:
            self._diarizer = DiarizationRegistry.create(
                self.config.diarization.backend, config=self.config.diarization)
        return self._diarizer

    @property
    def chunker(self):
        if self._chunker is None:
            self._chunker = ChunkingRegistry.create(
                self.config.chunking.strategy, config=self.config.chunking)
        return self._chunker

    def _get_contextual_processor(self) -> ContextualProcessor | None:
        """Get or create contextual processor."""
        if self._contextual is None:
            self._contextual = ContextualProcessor(config=self.config.generation)
        return self._contextual

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

    @timed
    def ingest(
        self,
        audio_path: Path | str,
        collection_name: str | None = None,
        metadata: dict | None = None,
        enable_diarization: bool = True,
        enable_contextual: bool | None = None,
        language: str | None = None,
    ) -> IngestionResult:
        """Ingest an audio file into the vector store."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        resolved_collection = collection_name or self.config.retrieval.collection_name
        base_metadata = metadata or {}
        use_contextual = enable_contextual if enable_contextual is not None else self.config.contextual.enabled

        logger.info(f"Ingesting: {audio_path.name} → {resolved_collection} (contextual={use_contextual})")

        try:
            # Step 1: Transcribe
            logger.info("Step 1/6: Transcribing audio...")
            self.resource_manager.ensure_vram(self.asr.vram_required)
            segments, words = self.asr.transcribe_with_words(audio_path, language=language)

            if not segments:
                return IngestionResult(
                    audio_path=str(audio_path), collection_name=resolved_collection,
                    num_segments=0, num_chunks=0, duration_seconds=0,
                    speakers=[], language=None, contextualized=False, metadata=base_metadata)

            detected_language = segments[0].language if segments else None

            # Step 2: Diarization
            speakers = []
            if enable_diarization and words:
                logger.info("Step 2/6: Diarizing speakers...")
                self.resource_manager.ensure_vram(self.diarizer.vram_required)
                diarization_segments = self.diarizer.diarize(audio_path)

                logger.info("Step 3/6: Aligning transcription to speakers...")
                aligned_words = align_words_to_speakers(words, diarization_segments)
                segments = build_speaker_transcript(aligned_words)
                speakers = list(set(s.speaker for s in segments if s.speaker))
            else:
                logger.info("Step 2/6: Skipping diarization")
                logger.info("Step 3/6: Skipping alignment")

            # Step 4: Chunk
            logger.info("Step 4/6: Chunking transcript...")
            chunks = self.chunker.chunk(segments)

            if not chunks:
                return IngestionResult(
                    audio_path=str(audio_path), collection_name=resolved_collection,
                    num_segments=len(segments), num_chunks=0,
                    duration_seconds=segments[-1].end if segments else 0,
                    speakers=speakers, language=detected_language,
                    contextualized=False, metadata=base_metadata)

            # Step 5: Contextual processing
            contextualized = False
            if use_contextual:
                logger.info("Step 5/6: Adding context to chunks...")
                processor = self._get_contextual_processor()
                if processor and processor.is_available:
                    chunks = processor.process_chunks(
                        chunks, 
                        window_size=self.config.contextual.window_size,
                        show_progress=True)
                    contextualized = True
                else:
                    logger.warning("Contextual processor not available, skipping")
            else:
                logger.info("Step 5/6: Skipping contextual processing")

            # Step 6: Embed and store
            logger.info("Step 6/6: Embedding and storing...")
            self.resource_manager.ensure_vram(self.embedder.vram_required)

            texts = [chunk.text for chunk in chunks]
            embeddings = self.embedder.embed(texts)

            for chunk in chunks:
                if chunk.metadata is None:
                    chunk.metadata = {}
                chunk.metadata["source"] = str(audio_path)
                chunk.metadata["source_filename"] = audio_path.name
                chunk.metadata.update(base_metadata)

            self.retriever.add(chunks, embeddings, collection_name=resolved_collection)

            duration = segments[-1].end if segments else 0

            logger.info(f"Ingestion complete: {len(chunks)} chunks, {len(speakers)} speakers, "
                       f"{duration:.1f}s, contextual={contextualized}")

            return IngestionResult(
                audio_path=str(audio_path), collection_name=resolved_collection,
                num_segments=len(segments), num_chunks=len(chunks),
                duration_seconds=duration, speakers=speakers,
                language=detected_language, contextualized=contextualized,
                metadata=base_metadata)

        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Ingestion failed for {audio_path}: {e}")
            raise PipelineError(f"Ingestion failed: {e}") from e

    def unload_all(self) -> None:
        if self._asr and self._asr.is_loaded:
            self._asr.unload()
        if self._diarizer and self._diarizer.is_loaded:
            self._diarizer.unload()
        if self._embedder and self._embedder.is_loaded:
            self._embedder.unload()
        self.resource_manager.unload_all()
        logger.info("All pipeline models unloaded")
