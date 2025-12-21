"""TTS registry and base configuration."""

from audio_rag.core import Registry, BaseTTS

# TTS Registry - all TTS backends register here
TTSRegistry = Registry[BaseTTS]("tts")
