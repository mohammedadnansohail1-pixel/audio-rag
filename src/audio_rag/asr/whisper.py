"""Faster Whisper ASR implementation."""

import gc
from pathlib import Path

from audio_rag.asr.base import ASRRegistry
from audio_rag.core import BaseASR, TranscriptSegment, ASRError
from audio_rag.config import ASRConfig
from audio_rag.utils import get_logger, timed, require_loaded

logger = get_logger(__name__)

# VRAM estimates by model size (GB)
VRAM_ESTIMATES = {
    "tiny": 1.0,
    "base": 1.5,
    "small": 2.5,
    "medium": 5.0,
    "large-v2": 6.0,
    "large-v3": 6.0,
}


@ASRRegistry.register("faster-whisper")
class FasterWhisperASR(BaseASR):
    """Faster Whisper ASR backend using CTranslate2."""
    
    def __init__(self, config: ASRConfig):
        self.config = config
        self._model = None
        self._device = self._resolve_device(config.device)
        logger.info(
            f"FasterWhisperASR initialized: model={config.model_size}, "
            f"device={self._device}, compute={config.compute_type}"
        )
    
    def _resolve_device(self, device: str) -> str:
        """Resolve 'auto' to actual device."""
        if device != "auto":
            return device
        
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    
    def load(self) -> None:
        """Load Whisper model into memory."""
        if self._model is not None:
            logger.debug("Model already loaded")
            return
        
        try:
            from faster_whisper import WhisperModel
            
            logger.info(f"Loading Whisper {self.config.model_size} on {self._device}...")
            self._model = WhisperModel(
                self.config.model_size,
                device=self._device,
                compute_type=self.config.compute_type,
            )
            logger.info("Whisper model loaded successfully")
            
        except Exception as e:
            raise ASRError(f"Failed to load Whisper model: {e}")
    
    def unload(self) -> None:
        """Unload model and free memory."""
        if self._model is None:
            return
        
        logger.info("Unloading Whisper model...")
        del self._model
        self._model = None
        gc.collect()
        
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        logger.info("Whisper model unloaded")
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
    
    @property
    def vram_required(self) -> float:
        """VRAM required in GB."""
        return VRAM_ESTIMATES.get(self.config.model_size, 6.0)
    
    @timed
    @require_loaded
    def transcribe(
        self, audio_path: Path, language: str | None = None
    ) -> list[TranscriptSegment]:
        """Transcribe audio file to segments.
        
        Args:
            audio_path: Path to audio file
            language: Language code (None for auto-detect)
            
        Returns:
            List of transcript segments with timestamps
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise ASRError(f"Audio file not found: {audio_path}")
        
        # Use config language if not specified
        lang = language or self.config.language
        
        try:
            segments_iter, info = self._model.transcribe(
                str(audio_path),
                language=lang,
                vad_filter=self.config.vad_filter,
                vad_parameters={"threshold": self.config.vad_threshold},
                word_timestamps=True,  # Needed for alignment
            )
            
            detected_lang = info.language
            logger.info(f"Detected language: {detected_lang} (prob={info.language_probability:.2f})")
            
            # Convert to TranscriptSegment
            segments = []
            for seg in segments_iter:
                segments.append(
                    TranscriptSegment(
                        text=seg.text.strip(),
                        start=seg.start,
                        end=seg.end,
                        speaker=None,  # Filled by diarization
                        confidence=seg.avg_logprob,
                        language=detected_lang,
                    )
                )
            
            logger.info(f"Transcribed {len(segments)} segments from {audio_path.name}")
            return segments
            
        except Exception as e:
            raise ASRError(f"Transcription failed: {e}")
    
    def transcribe_with_words(
        self, audio_path: Path, language: str | None = None
    ) -> tuple[list[TranscriptSegment], list[dict]]:
        """Transcribe with word-level timestamps for alignment.
        
        Returns:
            Tuple of (segments, words) where words is list of
            {"word": str, "start": float, "end": float}
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise ASRError(f"Audio file not found: {audio_path}")
        
        if not self.is_loaded:
            self.load()
        
        lang = language or self.config.language
        
        try:
            segments_iter, info = self._model.transcribe(
                str(audio_path),
                language=lang,
                vad_filter=self.config.vad_filter,
                vad_parameters={"threshold": self.config.vad_threshold},
                word_timestamps=True,
            )
            
            detected_lang = info.language
            segments = []
            all_words = []
            
            for seg in segments_iter:
                segments.append(
                    TranscriptSegment(
                        text=seg.text.strip(),
                        start=seg.start,
                        end=seg.end,
                        speaker=None,
                        confidence=seg.avg_logprob,
                        language=detected_lang,
                    )
                )
                
                # Extract word-level timestamps
                if seg.words:
                    for word in seg.words:
                        all_words.append({
                            "word": word.word.strip(),
                            "start": word.start,
                            "end": word.end,
                        })
            
            logger.info(f"Transcribed {len(segments)} segments, {len(all_words)} words")
            return segments, all_words
            
        except Exception as e:
            raise ASRError(f"Transcription with words failed: {e}")
