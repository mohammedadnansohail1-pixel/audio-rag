"""Edge TTS implementation (Microsoft Edge online TTS)."""

import asyncio
from pathlib import Path

from audio_rag.tts.base import TTSRegistry
from audio_rag.core import BaseTTS, TTSError
from audio_rag.config import TTSConfig
from audio_rag.utils import get_logger, timed

logger = get_logger(__name__)

# Common Edge TTS voices by language
EDGE_VOICES = {
    "en": "en-US-AriaNeural",
    "en-US": "en-US-AriaNeural",
    "en-GB": "en-GB-SoniaNeural",
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "it": "it-IT-ElsaNeural",
    "pt": "pt-BR-FranciscaNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "hi": "hi-IN-SwaraNeural",
    "ar": "ar-SA-ZariyahNeural",
    "ru": "ru-RU-SvetlanaNeural",
}


@TTSRegistry.register("edge-tts")
class EdgeTTS(BaseTTS):
    """Edge TTS backend using Microsoft Edge's online TTS.
    
    Pros: 100+ languages, zero VRAM, high quality
    Cons: Requires internet, API-based
    """
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self._voice = config.model if config.model.endswith("Neural") else EDGE_VOICES.get("en")
        logger.info(f"EdgeTTS initialized: voice={self._voice}")
    
    def load(self) -> None:
        """No loading needed for Edge TTS."""
        logger.debug("EdgeTTS: No model to load (API-based)")
    
    def unload(self) -> None:
        """No unloading needed for Edge TTS."""
        logger.debug("EdgeTTS: No model to unload (API-based)")
    
    @property
    def is_loaded(self) -> bool:
        """Always ready (API-based)."""
        return True
    
    @property
    def vram_required(self) -> float:
        """Zero VRAM required."""
        return 0.0
    
    @timed
    def synthesize(
        self, text: str, output_path: Path, language: str | None = None
    ) -> Path:
        """Synthesize text to audio file.
        
        Args:
            text: Text to synthesize
            output_path: Path for output audio file
            language: Language code (selects appropriate voice)
            
        Returns:
            Path to generated audio file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Select voice based on language
        voice = self._voice
        if language and language in EDGE_VOICES:
            voice = EDGE_VOICES[language]
        
        try:
            # Run async function in sync context
            asyncio.run(self._synthesize_async(text, output_path, voice))
            logger.info(f"Synthesized {len(text)} chars to {output_path.name}")
            return output_path
            
        except Exception as e:
            raise TTSError(f"Edge TTS synthesis failed: {e}")
    
    async def _synthesize_async(self, text: str, output_path: Path, voice: str) -> None:
        """Async synthesis implementation."""
        import edge_tts
        
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(output_path))
    
    def get_available_voices(self, language: str | None = None) -> list[str]:
        """Get available voices, optionally filtered by language."""
        if language:
            return [v for k, v in EDGE_VOICES.items() if k.startswith(language)]
        return list(EDGE_VOICES.values())
