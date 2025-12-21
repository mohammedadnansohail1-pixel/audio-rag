"""Text-to-speech module."""

from audio_rag.tts.base import TTSRegistry
from audio_rag.tts.edge import EdgeTTS
from audio_rag.tts.piper import PiperTTS

__all__ = [
    "TTSRegistry",
    "EdgeTTS",
    "PiperTTS",
]
