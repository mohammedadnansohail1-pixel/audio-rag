"""Fixed-size chunking strategy."""

from audio_rag.chunking.base import ChunkingRegistry
from audio_rag.core import BaseChunker, TranscriptSegment, AudioChunk
from audio_rag.config import ChunkingConfig
from audio_rag.utils import get_logger

logger = get_logger(__name__)


def estimate_tokens(text: str) -> int:
    """Estimate token count (rough: ~4 chars per token)."""
    return len(text) // 4


@ChunkingRegistry.register("fixed")
class FixedSizeChunker(BaseChunker):
    """Fixed-size chunking with token limits.
    
    Simple chunking that ignores speaker boundaries.
    Useful for monologue content or when speaker info isn't needed.
    """
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.max_tokens = config.max_tokens
        self.overlap_tokens = config.overlap_tokens
        logger.info(f"FixedSizeChunker: max={self.max_tokens}, overlap={self.overlap_tokens}")
    
    def chunk(self, segments: list[TranscriptSegment]) -> list[AudioChunk]:
        """Convert transcript segments into fixed-size chunks.
        
        Args:
            segments: Transcript segments
            
        Returns:
            List of AudioChunks
        """
        if not segments:
            return []
        
        # Flatten all text with timing info
        words_with_timing = []
        for seg in segments:
            words = seg.text.split()
            if not words:
                continue
            
            # Distribute timing across words
            duration = seg.end - seg.start
            word_duration = duration / len(words) if words else 0
            
            for i, word in enumerate(words):
                words_with_timing.append({
                    "word": word,
                    "start": seg.start + i * word_duration,
                    "end": seg.start + (i + 1) * word_duration,
                    "speaker": seg.speaker,
                    "language": seg.language,
                })
        
        # Build chunks
        chunks = []
        current_words = []
        current_tokens = 0
        chunk_start = words_with_timing[0]["start"] if words_with_timing else 0
        
        for word_info in words_with_timing:
            word_tokens = estimate_tokens(word_info["word"])
            
            if current_tokens + word_tokens > self.max_tokens and current_words:
                # Save current chunk
                chunks.append(
                    AudioChunk(
                        text=" ".join(w["word"] for w in current_words),
                        start=chunk_start,
                        end=current_words[-1]["end"],
                        speaker=self._majority_speaker(current_words),
                        metadata={
                            "word_count": len(current_words),
                            "duration": current_words[-1]["end"] - chunk_start,
                        },
                    )
                )
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_words) - self._words_for_tokens(self.overlap_tokens, current_words))
                current_words = current_words[overlap_start:]
                current_tokens = sum(estimate_tokens(w["word"]) for w in current_words)
                chunk_start = current_words[0]["start"] if current_words else word_info["start"]
            
            current_words.append(word_info)
            current_tokens += word_tokens
        
        # Last chunk
        if current_words:
            chunks.append(
                AudioChunk(
                    text=" ".join(w["word"] for w in current_words),
                    start=chunk_start,
                    end=current_words[-1]["end"],
                    speaker=self._majority_speaker(current_words),
                    metadata={
                        "word_count": len(current_words),
                        "duration": current_words[-1]["end"] - chunk_start,
                    },
                )
            )
        
        logger.info(f"Created {len(chunks)} fixed-size chunks")
        return chunks
    
    def _majority_speaker(self, words: list[dict]) -> str | None:
        """Get most common speaker in word list."""
        speakers = [w["speaker"] for w in words if w.get("speaker")]
        if not speakers:
            return None
        return max(set(speakers), key=speakers.count)
    
    def _words_for_tokens(self, token_count: int, words: list[dict]) -> int:
        """Estimate how many words fit in token count."""
        count = 0
        tokens = 0
        for word in reversed(words):
            tokens += estimate_tokens(word["word"])
            if tokens > token_count:
                break
            count += 1
        return count
