"""Contextual processor - generates context for chunks using LLM."""

import httpx
from typing import Any

from audio_rag.core import AudioChunk
from audio_rag.config import GenerationConfig
from audio_rag.utils import get_logger

logger = get_logger(__name__)


CONTEXT_SYSTEM_PROMPT = """You are an assistant that generates brief contextual descriptions for transcript excerpts.
Your task is to write a 1-2 sentence context that explains:
- What topic is being discussed
- How this excerpt fits in the broader lecture/conversation
Be concise and factual."""

CONTEXT_PROMPT_TEMPLATE = """Here is an excerpt from an audio transcript:

SPEAKER: {speaker}
TIMESTAMP: {start:.1f}s - {end:.1f}s
TEXT: {text}

{surrounding_context}

Write a brief 1-2 sentence context that situates this excerpt. Start with "This excerpt" or "In this section"."""


class ContextualProcessor:
    """Adds LLM-generated context to chunks before embedding.
    
    This implements Anthropic's Contextual Retrieval technique:
    - Prepends each chunk with context about where it fits in the document
    - Reduces retrieval failures by 49% according to Anthropic research
    - Context is embedded along with the chunk text
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self.model = config.model
        self.timeout = config.timeout
        self._client = httpx.Client(timeout=self.timeout)
        self._is_available = self._check_availability()
        
        if self._is_available:
            logger.info(f"ContextualProcessor ready: model={self.model}")
        else:
            logger.warning("ContextualProcessor unavailable - chunks will not have context")

    def _check_availability(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = self._client.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                return False
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            return any(self.model in name for name in model_names)
        except Exception:
            return False

    @property
    def is_available(self) -> bool:
        return self._is_available

    def generate_context(
        self, 
        chunk: AudioChunk,
        preceding_text: str | None = None,
        following_text: str | None = None,
    ) -> str:
        """Generate context for a single chunk.
        
        Args:
            chunk: The chunk to contextualize
            preceding_text: Text from previous chunks (optional)
            following_text: Text from following chunks (optional)
            
        Returns:
            Generated context string
        """
        if not self._is_available:
            return ""

        # Build surrounding context hint
        surrounding = ""
        if preceding_text:
            surrounding += f"PRECEDING TEXT: ...{preceding_text[-200:]}\n"
        if following_text:
            surrounding += f"FOLLOWING TEXT: {following_text[:200]}...\n"

        prompt = CONTEXT_PROMPT_TEMPLATE.format(
            speaker=chunk.speaker or "Unknown",
            start=chunk.start,
            end=chunk.end,
            text=chunk.text[:500],  # Truncate long chunks
            surrounding_context=surrounding,
        )

        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "system": CONTEXT_SYSTEM_PROMPT,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower for factual context
                    "num_predict": 100,  # Short context
                },
            }

            response = self._client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()

            result = response.json()
            context = result.get("response", "").strip()
            
            # Clean up context
            if context and not context.endswith('.'):
                context += '.'
                
            return context

        except Exception as e:
            logger.warning(f"Context generation failed: {e}")
            return ""

    def process_chunks(
        self,
        chunks: list[AudioChunk],
        window_size: int = 1,
        show_progress: bool = True,
    ) -> list[AudioChunk]:
        """Add context to all chunks.
        
        Args:
            chunks: List of chunks to process
            window_size: Number of surrounding chunks to consider
            show_progress: Whether to show progress
            
        Returns:
            New list of chunks with context prepended to text
        """
        if not self._is_available:
            logger.warning("ContextualProcessor not available, returning original chunks")
            return chunks

        if not chunks:
            return chunks

        processed = []
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            if show_progress and (i + 1) % 10 == 0:
                logger.info(f"Processing context: {i + 1}/{total}")

            # Get surrounding text
            preceding = None
            following = None
            
            if window_size > 0:
                if i > 0:
                    preceding_chunks = chunks[max(0, i - window_size):i]
                    preceding = " ".join(c.text for c in preceding_chunks)
                if i < len(chunks) - 1:
                    following_chunks = chunks[i + 1:min(len(chunks), i + 1 + window_size)]
                    following = " ".join(c.text for c in following_chunks)

            # Generate context
            context = self.generate_context(chunk, preceding, following)

            # Create new chunk with context prepended
            if context:
                contextualized_text = f"[Context: {context}]\n{chunk.text}"
            else:
                contextualized_text = chunk.text

            new_chunk = AudioChunk(
                text=contextualized_text,
                start=chunk.start,
                end=chunk.end,
                speaker=chunk.speaker,
                metadata={
                    **(chunk.metadata or {}),
                    "original_text": chunk.text,
                    "context": context,
                    "contextualized": bool(context),
                },
            )
            processed.append(new_chunk)

        contextualized_count = sum(1 for c in processed if c.metadata.get("contextualized"))
        logger.info(f"Contextualized {contextualized_count}/{total} chunks")

        return processed

    def __del__(self):
        try:
            self._client.close()
        except Exception:
            pass
