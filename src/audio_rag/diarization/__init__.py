"""Speaker diarization module."""

from audio_rag.diarization.base import DiarizationRegistry
from audio_rag.diarization.pyannote import PyAnnoteDiarizer

__all__ = [
    "DiarizationRegistry",
    "PyAnnoteDiarizer",
]
