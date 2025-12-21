"""Retrieval registry and base configuration."""

from audio_rag.core import Registry, BaseRetriever

# Retrieval Registry - all retrieval backends register here
RetrievalRegistry = Registry[BaseRetriever]("retrieval")
