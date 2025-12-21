"""Audio ingestion pipeline - Audio → Chunks → Vectors."""

from pathlib import Path
from dataclasses import dataclass

from audio_rag.core import TranscriptSegment, AudioChunk
from audio_rag.asr import ASRRegistry
from audio_rag.diarization import DiarizationRegistry
from audio_rag.alignment import align_words_to_speakers, build_speaker_transcript
from audio_rag.chunking import ChunkingRegistry
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
    num_segments: int
    num_chunks: int
    duration_seconds: float
    speakers: list[str]
    language: str | None


class IngestionPipeline:
    """Pipeline for ingesting audio into the vector store.
    
    Flow: Audio → ASR → Diarization → Alignment → Chunking → Embedding → Store
    """
    
    def __init__(self, config: AudioRAGConfig, resource_manager: ResourceManager | None = None):
        self.config = config
        self.resource_manager = resource_manager or ResourceManager(config.resources)
        
        # Lazy-loaded components
        self._asr = None
        self._diarizer = None
        self._chunker = None
        self._embedder = None
        self._retriever = None
        
        logger.info("IngestionPipeline initialized")
    
    @property
    def asr(self):
        """Lazy-load ASR."""
        if self._asr is None:
            self._asr = ASRRegistry.create(
                self.config.asr.backend,
                config=self.config.asr,
            )
        return self._asr
    
    @property
    def diarizer(self):
        """Lazy-load diarizer."""
        if self._diarizer is None:
            self._diarizer = DiarizationRegistry.create(
                self.config.diarization.backend,
                config=self.config.diarization,
            )
        return self._diarizer
    
    @property
    def chunker(self):
        """Lazy-load chunker."""
        if self._chunker is None:
            self._chunker = ChunkingRegistry.create(
                self.config.chunking.strategy,
                config=self.config.chunking,
            )
        return self._chunker
    
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
            # Get embedding dimension (load embedder if needed)
            if not self.embedder.is_loaded:
                self.embedder.load()
            
            self._retriever = RetrievalRegistry.create(
                self.config.retrieval.backend,
                config=self.config.retrieval,
                embedding_dim=self.embedder.dimension,
            )
        return self._retriever
    
    @timed
    def ingest(
        self,
        audio_path: Path | str,
        enable_diarization: bool = True,
        language: str | None = None,
    ) -> IngestionResult:
        """Ingest an audio file into the vector store.
        
        Args:
            audio_path: Path to audio file
            enable_diarization: Whether to perform speaker diarization
            language: Language code (None for auto-detect)
            
        Returns:
            IngestionResult with statistics
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Ingesting: {audio_path.name}")
        
        # Step 1: Transcribe with word timestamps
        logger.info("Step 1/5: Transcribing audio...")
        self.resource_manager.ensure_vram(self.asr.vram_required)
        segments, words = self.asr.transcribe_with_words(audio_path, language=language)
        
        if not segments:
            logger.warning("No speech detected in audio")
            return IngestionResult(
                audio_path=str(audio_path),
                num_segments=0,
                num_chunks=0,
                duration_seconds=0,
                speakers=[],
                language=None,
            )
        
        detected_language = segments[0].language if segments else None
        
        # Step 2: Diarization (optional)
        speakers = []
        if enable_diarization and words:
            logger.info("Step 2/5: Diarizing speakers...")
            self.resource_manager.ensure_vram(self.diarizer.vram_required)
            diarization_segments = self.diarizer.diarize(audio_path)
            
            # Step 3: Align words to speakers
            logger.info("Step 3/5: Aligning transcription to speakers...")
            aligned_words = align_words_to_speakers(words, diarization_segments)
            
            # Build speaker-attributed transcript
            segments = build_speaker_transcript(aligned_words)
            speakers = list(set(s.speaker for s in segments if s.speaker))
        else:
            logger.info("Step 2/5: Skipping diarization")
            logger.info("Step 3/5: Skipping alignment")
        
        # Step 4: Chunk
        logger.info("Step 4/5: Chunking transcript...")
        chunks = self.chunker.chunk(segments)
        
        if not chunks:
            logger.warning("No chunks created from transcript")
            return IngestionResult(
                audio_path=str(audio_path),
                num_segments=len(segments),
                num_chunks=0,
                duration_seconds=segments[-1].end if segments else 0,
                speakers=speakers,
                language=detected_language,
            )
        
        # Step 5: Embed and store
        logger.info("Step 5/5: Embedding and storing...")
        self.resource_manager.ensure_vram(self.embedder.vram_required)
        
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed(texts)
        
        # Add source metadata
        for chunk in chunks:
            if chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata["source"] = str(audio_path)
        
        self.retriever.add(chunks, embeddings)
        
        duration = segments[-1].end if segments else 0
        
        logger.info(
            f"Ingestion complete: {len(chunks)} chunks from {len(segments)} segments, "
            f"{len(speakers)} speakers, {duration:.1f}s duration"
        )
        
        return IngestionResult(
            audio_path=str(audio_path),
            num_segments=len(segments),
            num_chunks=len(chunks),
            duration_seconds=duration,
            speakers=speakers,
            language=detected_language,
        )
    
    def unload_all(self) -> None:
        """Unload all models to free memory."""
        if self._asr and self._asr.is_loaded:
            self._asr.unload()
        if self._diarizer and self._diarizer.is_loaded:
            self._diarizer.unload()
        if self._embedder and self._embedder.is_loaded:
            self._embedder.unload()
        
        self.resource_manager.unload_all()
        logger.info("All pipeline models unloaded")
