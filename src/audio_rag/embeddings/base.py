"""Embeddings registry and base configuration."""

from audio_rag.core import Registry, BaseEmbedder

# Embeddings Registry - all embedding backends register here
EmbeddingsRegistry = Registry[BaseEmbedder]("embeddings")
