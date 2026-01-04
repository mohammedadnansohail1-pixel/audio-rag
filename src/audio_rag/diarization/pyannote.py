"""PyAnnote speaker diarization implementation."""

import gc
from pathlib import Path

from audio_rag.diarization.base import DiarizationRegistry
from audio_rag.core import BaseDiarizer, TranscriptSegment, DiarizationError
from audio_rag.config import DiarizationConfig
from audio_rag.utils import get_logger, timed, require_loaded

logger = get_logger(__name__)

# VRAM estimate for PyAnnote
VRAM_ESTIMATE = 2.5  # GB


@DiarizationRegistry.register("pyannote")
class PyAnnoteDiarizer(BaseDiarizer):
    """PyAnnote speaker diarization backend."""
    
    def __init__(self, config: DiarizationConfig):
        self.config = config
        self._pipeline = None
        self._device = self._resolve_device(config.device)
        logger.info(
            f"PyAnnoteDiarizer initialized: model={config.model}, device={self._device}"
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
        """Load diarization pipeline into memory."""
        if self._pipeline is not None:
            logger.debug("Pipeline already loaded")
            return
        
        try:
            import torch
            from pyannote.audio import Pipeline
            
            logger.info(f"Loading PyAnnote pipeline: {self.config.model}...")
            
            self._pipeline = Pipeline.from_pretrained(
                self.config.model,
                token=self._get_hf_token(),
            )
            
            # Move to device
            if self._device == "cuda" and torch.cuda.is_available():
                self._pipeline = self._pipeline.to(torch.device("cuda"))
            
            logger.info("PyAnnote pipeline loaded successfully")
            
        except Exception as e:
            raise DiarizationError(f"Failed to load PyAnnote pipeline: {e}")
    
    def _get_hf_token(self) -> str | None:
        """Get HuggingFace token from environment."""
        import os
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not token:
            logger.warning(
                "No HuggingFace token found. Set HF_TOKEN environment variable. "
                "PyAnnote models require accepting license at huggingface.co"
            )
        return token
    
    def unload(self) -> None:
        """Unload pipeline and free memory."""
        if self._pipeline is None:
            return
        
        logger.info("Unloading PyAnnote pipeline...")
        del self._pipeline
        self._pipeline = None
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        logger.info("PyAnnote pipeline unloaded")
    
    @property
    def is_loaded(self) -> bool:
        """Check if pipeline is loaded."""
        return self._pipeline is not None
    
    @property
    def vram_required(self) -> float:
        """VRAM required in GB."""
        return VRAM_ESTIMATE
    
    @timed
    @require_loaded
    def diarize(
        self,
        audio_path: Path,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[TranscriptSegment]:
        """Identify speaker segments in audio.
        
        Args:
            audio_path: Path to audio file
            min_speakers: Minimum expected speakers (optional)
            max_speakers: Maximum expected speakers (optional)
            
        Returns:
            List of segments with speaker labels (no text, just timing)
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise DiarizationError(f"Audio file not found: {audio_path}")
        
        # Use config values if not specified
        min_spk = min_speakers or self.config.min_speakers
        max_spk = max_speakers or self.config.max_speakers
        
        try:
            # Build diarization parameters
            params = {}
            if min_spk is not None:
                params["min_speakers"] = min_spk
            if max_spk is not None:
                params["max_speakers"] = max_spk
            
            logger.info(f"Diarizing {audio_path.name}...")
            diarization = self._pipeline(str(audio_path), **params)
            
            # Convert to TranscriptSegment
            segments = []
            for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
                segments.append(
                    TranscriptSegment(
                        text="",  # Filled by alignment
                        start=turn.start,
                        end=turn.end,
                        speaker=speaker,
                        confidence=None,
                        language=None,
                    )
                )
            
            # Count unique speakers
            speakers = set(seg.speaker for seg in segments)
            logger.info(f"Diarization complete: {len(segments)} turns, {len(speakers)} speakers")
            
            return segments
            
        except Exception as e:
            raise DiarizationError(f"Diarization failed: {e}")
    
    def get_speaker_timeline(
        self, audio_path: Path, min_speakers: int | None = None, max_speakers: int | None = None
    ):
        """Get raw pyannote Annotation object for advanced use.
        
        Returns the native pyannote annotation for use with alignment.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise DiarizationError(f"Audio file not found: {audio_path}")
        
        if not self.is_loaded:
            self.load()
        
        min_spk = min_speakers or self.config.min_speakers
        max_spk = max_speakers or self.config.max_speakers
        
        params = {}
        if min_spk is not None:
            params["min_speakers"] = min_spk
        if max_spk is not None:
            params["max_speakers"] = max_spk
        
        try:
            return self._pipeline(str(audio_path), **params)
        except Exception as e:
            raise DiarizationError(f"Diarization failed: {e}")
