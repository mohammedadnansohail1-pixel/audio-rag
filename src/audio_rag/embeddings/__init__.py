"""Embeddings module."""

from audio_rag.embeddings.base import EmbeddingsRegistry
from audio_rag.embeddings.bge import BGEM3Embedder

__all__ = [
    "EmbeddingsRegistry",
    "BGEM3Embedder",
]
