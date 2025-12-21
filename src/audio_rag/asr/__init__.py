"""ASR (Automatic Speech Recognition) module."""

from audio_rag.asr.base import ASRRegistry
from audio_rag.asr.whisper import FasterWhisperASR

__all__ = [
    "ASRRegistry",
    "FasterWhisperASR",
]
