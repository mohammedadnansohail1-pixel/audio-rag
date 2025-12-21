"""Diarization registry and base configuration."""

from audio_rag.core import Registry, BaseDiarizer

# Diarization Registry - all diarization backends register here
DiarizationRegistry = Registry[BaseDiarizer]("diarization")
