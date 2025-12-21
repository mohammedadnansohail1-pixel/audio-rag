"""Chunking registry and base configuration."""

from audio_rag.core import Registry, BaseChunker

# Chunking Registry - all chunking strategies register here
ChunkingRegistry = Registry[BaseChunker]("chunking")
