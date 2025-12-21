"""ASR registry and base configuration."""

from audio_rag.core import Registry, BaseASR

# ASR Registry - all ASR backends register here
ASRRegistry = Registry[BaseASR]("asr")
