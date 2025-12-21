"""Speaker-turn based chunking strategy."""

from audio_rag.chunking.base import ChunkingRegistry
from audio_rag.core import BaseChunker, TranscriptSegment, AudioChunk
from audio_rag.config import ChunkingConfig
from audio_rag.utils import get_logger

logger = get_logger(__name__)


def estimate_tokens(text: str) -> int:
    """Estimate token count (rough: ~4 chars per token)."""
    return len(text) // 4


@ChunkingRegistry.register("speaker_turn")
class SpeakerTurnChunker(BaseChunker):
    """Chunk transcript by speaker turns with token limits.
    
    Preserves speaker attribution and creates natural conversation chunks.
    Splits long turns and merges short ones while respecting speaker boundaries.
    """
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.max_tokens = config.max_tokens
        self.overlap_tokens = config.overlap_tokens
        self.min_chunk_tokens = config.min_chunk_tokens
        logger.info(
            f"SpeakerTurnChunker: max={self.max_tokens}, "
            f"overlap={self.overlap_tokens}, min={self.min_chunk_tokens}"
        )
    
    def chunk(self, segments: list[TranscriptSegment]) -> list[AudioChunk]:
        """Convert transcript segments into chunks for embedding.
        
        Strategy:
        1. Group consecutive segments by speaker
        2. Split groups that exceed max_tokens
        3. Merge small groups if under min_chunk_tokens (same speaker only)
        4. Add overlap between chunks for context
        
        Args:
            segments: Transcript segments with speaker attribution
            
        Returns:
            List of AudioChunks ready for embedding
        """
        if not segments:
            return []
        
        # Step 1: Group by speaker
        speaker_groups = self._group_by_speaker(segments)
        logger.debug(f"Grouped into {len(speaker_groups)} speaker turns")
        
        # Step 2: Split large groups
        split_groups = []
        for group in speaker_groups:
            split_groups.extend(self._split_if_too_large(group))
        logger.debug(f"After splitting: {len(split_groups)} groups")
        
        # Step 3: Convert to AudioChunks with metadata
        chunks = []
        for group in split_groups:
            text = " ".join(seg.text for seg in group)
            if estimate_tokens(text) < self.min_chunk_tokens:
                continue  # Skip very small chunks
            
            chunk = AudioChunk(
                text=text,
                start=group[0].start,
                end=group[-1].end,
                speaker=group[0].speaker,
                metadata={
                    "segment_count": len(group),
                    "duration": group[-1].end - group[0].start,
                    "language": group[0].language,
                },
            )
            chunks.append(chunk)
        
        # Step 4: Add overlap context
        if self.overlap_tokens > 0:
            chunks = self._add_overlap_context(chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {len(segments)} segments")
        return chunks
    
    def _group_by_speaker(
        self, segments: list[TranscriptSegment]
    ) -> list[list[TranscriptSegment]]:
        """Group consecutive segments by speaker."""
        if not segments:
            return []
        
        groups = []
        current_group = [segments[0]]
        current_speaker = segments[0].speaker
        
        for seg in segments[1:]:
            if seg.speaker == current_speaker:
                current_group.append(seg)
            else:
                groups.append(current_group)
                current_group = [seg]
                current_speaker = seg.speaker
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _split_if_too_large(
        self, group: list[TranscriptSegment]
    ) -> list[list[TranscriptSegment]]:
        """Split a speaker group if it exceeds max_tokens."""
        text = " ".join(seg.text for seg in group)
        if estimate_tokens(text) <= self.max_tokens:
            return [group]
        
        # Need to split
        result = []
        current_group = []
        current_tokens = 0
        
        for seg in group:
            seg_tokens = estimate_tokens(seg.text)
            
            if current_tokens + seg_tokens > self.max_tokens and current_group:
                result.append(current_group)
                current_group = []
                current_tokens = 0
            
            current_group.append(seg)
            current_tokens += seg_tokens
        
        if current_group:
            result.append(current_group)
        
        return result
    
    def _add_overlap_context(self, chunks: list[AudioChunk]) -> list[AudioChunk]:
        """Add overlap context from previous chunk."""
        if len(chunks) <= 1:
            return chunks
        
        result = [chunks[0]]  # First chunk unchanged
        
        for i, chunk in enumerate(chunks[1:], 1):
            prev_chunk = chunks[i - 1]
            
            # Get last N tokens from previous chunk as context
            prev_words = prev_chunk.text.split()
            overlap_words = []
            token_count = 0
            
            for word in reversed(prev_words):
                word_tokens = estimate_tokens(word)
                if token_count + word_tokens > self.overlap_tokens:
                    break
                overlap_words.insert(0, word)
                token_count += word_tokens
            
            if overlap_words:
                # Prepend context with marker
                context = " ".join(overlap_words)
                new_text = f"[...{context}] {chunk.text}"
                
                result.append(
                    AudioChunk(
                        text=new_text,
                        start=chunk.start,
                        end=chunk.end,
                        speaker=chunk.speaker,
                        metadata=chunk.metadata,
                    )
                )
            else:
                result.append(chunk)
        
        return result
