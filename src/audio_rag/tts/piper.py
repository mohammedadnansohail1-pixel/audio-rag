"""Piper TTS implementation (local, lightweight)."""

import subprocess
import shutil
from pathlib import Path

from audio_rag.tts.base import TTSRegistry
from audio_rag.core import BaseTTS, TTSError
from audio_rag.config import TTSConfig
from audio_rag.utils import get_logger, timed

logger = get_logger(__name__)


@TTSRegistry.register("piper")
class PiperTTS(BaseTTS):
    """Piper TTS backend - fast local CPU-based TTS.
    
    Pros: Fast, CPU-only, 30+ languages, works offline
    Cons: No voice cloning, requires model download
    """
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self.model = config.model
        self.sample_rate = config.sample_rate
        self._piper_path = shutil.which("piper")
        self._loaded = False
        logger.info(f"PiperTTS initialized: model={self.model}")
    
    def load(self) -> None:
        """Verify Piper is available."""
        if self._loaded:
            return
        
        if not self._piper_path:
            # Try to find piper in common locations
            possible_paths = [
                Path.home() / ".local/bin/piper",
                Path("/usr/local/bin/piper"),
                Path("/usr/bin/piper"),
            ]
            for p in possible_paths:
                if p.exists():
                    self._piper_path = str(p)
                    break
        
        if not self._piper_path:
            logger.warning(
                "Piper not found in PATH. Install with: "
                "pip install piper-tts or download from github.com/rhasspy/piper"
            )
        else:
            logger.info(f"Piper found at: {self._piper_path}")
        
        self._loaded = True
    
    def unload(self) -> None:
        """No unloading needed for Piper."""
        logger.debug("PiperTTS: No model to unload")
    
    @property
    def is_loaded(self) -> bool:
        """Check if Piper is available."""
        return self._loaded
    
    @property
    def vram_required(self) -> float:
        """Zero VRAM - CPU only."""
        return 0.0
    
    @timed
    def synthesize(
        self, text: str, output_path: Path, language: str | None = None
    ) -> Path:
        """Synthesize text to audio file.
        
        Args:
            text: Text to synthesize
            output_path: Path for output audio file
            language: Language code (ignored, use model for language)
            
        Returns:
            Path to generated audio file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self._loaded:
            self.load()
        
        # Try Python API first, fall back to CLI
        try:
            return self._synthesize_python(text, output_path)
        except ImportError:
            logger.debug("Piper Python API not available, trying CLI")
            return self._synthesize_cli(text, output_path)
    
    def _synthesize_python(self, text: str, output_path: Path) -> Path:
        """Synthesize using Piper Python API."""
        import wave
        from piper import PiperVoice
        
        voice = PiperVoice.load(self.model)
        
        # Generate audio
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            
            for audio_bytes in voice.synthesize_stream_raw(text):
                wav_file.writeframes(audio_bytes)
        
        logger.info(f"Synthesized {len(text)} chars to {output_path.name}")
        return output_path
    
    def _synthesize_cli(self, text: str, output_path: Path) -> Path:
        """Synthesize using Piper CLI."""
        if not self._piper_path:
            raise TTSError("Piper not installed. Run: pip install piper-tts")
        
        try:
            result = subprocess.run(
                [
                    self._piper_path,
                    "--model", self.model,
                    "--output_file", str(output_path),
                ],
                input=text,
                capture_output=True,
                text=True,
                check=True,
            )
            
            logger.info(f"Synthesized {len(text)} chars to {output_path.name}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            raise TTSError(f"Piper CLI failed: {e.stderr}")
        except FileNotFoundError:
            raise TTSError("Piper not found in PATH")
