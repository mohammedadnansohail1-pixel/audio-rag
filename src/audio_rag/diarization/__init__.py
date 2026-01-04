"""Speaker diarization module."""
from audio_rag.diarization.base import DiarizationRegistry
from audio_rag.diarization.pyannote import PyAnnoteDiarizer
from audio_rag.diarization.nemo import NemoDiarizer

__all__ = [
    "DiarizationRegistry",
    "PyAnnoteDiarizer",
    "NemoDiarizer",
]
