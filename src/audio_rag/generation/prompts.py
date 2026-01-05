"""RAG prompt templates for audio transcript Q&A."""

from audio_rag.core import RetrievalResult


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on audio transcript excerpts.

Guidelines:
- Answer based ONLY on the provided transcript context
- If the context doesn't contain enough information, say so
- Reference specific speakers and timestamps when relevant
- Be concise but complete
- If multiple speakers discuss the topic, synthesize their perspectives"""


RAG_PROMPT_TEMPLATE = """Based on the following transcript excerpts, answer the user's question.

TRANSCRIPT CONTEXT:
{context}

USER QUESTION: {query}

Provide a clear, accurate answer based on the transcript context above."""


def build_rag_prompt(query: str, results: list[RetrievalResult]) -> str:
    """Build RAG prompt from query and retrieved results."""
    if not results:
        return f"No context found.\n\nUSER QUESTION: {query}"
    
    context_parts = []
    for i, result in enumerate(results, 1):
        chunk = result.chunk
        speaker = chunk.speaker or "Unknown"
        start = _format_timestamp(chunk.start)
        end = _format_timestamp(chunk.end)
        
        context_parts.append(
            f"[Excerpt {i}] ({speaker}, {start} - {end})\n{chunk.text}"
        )
    
    context = "\n\n".join(context_parts)
    return RAG_PROMPT_TEMPLATE.format(context=context, query=query)


def _format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
