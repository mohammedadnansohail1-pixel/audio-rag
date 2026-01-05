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
        if not segments:
            return []

        speaker_groups = self._group_by_speaker(segments)
        logger.debug(f"Grouped into {len(speaker_groups)} speaker turns")

        split_groups = []
        for group in speaker_groups:
            split_groups.extend(self._split_if_too_large(group))
        logger.debug(f"After splitting: {len(split_groups)} groups")

        merged_groups = self._merge_small_groups(split_groups)
        logger.debug(f"After merging: {len(merged_groups)} groups")

        chunks = []
        for group in merged_groups:
            text = " ".join(seg.text for seg in group)
            tokens = estimate_tokens(text)
            if not text.strip():
                continue
            chunk = AudioChunk(
                text=text,
                start=group[0].start,
                end=group[-1].end,
                speaker=group[0].speaker,
                metadata={
                    "segment_count": len(group),
                    "duration": group[-1].end - group[0].start,
                    "language": group[0].language,
                    "token_estimate": tokens,
                },
            )
            chunks.append(chunk)

        if self.overlap_tokens > 0:
            chunks = self._add_overlap_context(chunks)

        logger.info(f"Created {len(chunks)} chunks from {len(segments)} segments")
        return chunks

    def _group_by_speaker(self, segments: list[TranscriptSegment]) -> list[list[TranscriptSegment]]:
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

    def _split_if_too_large(self, group: list[TranscriptSegment]) -> list[list[TranscriptSegment]]:
        text = " ".join(seg.text for seg in group)
        if estimate_tokens(text) <= self.max_tokens:
            return [group]
        result = []
        current_group = []
        current_tokens = 0
        for seg in group:
            seg_tokens = estimate_tokens(seg.text)
            if seg_tokens > self.max_tokens and not current_group:
                result.append([seg])
                continue
            if current_tokens + seg_tokens > self.max_tokens and current_group:
                result.append(current_group)
                current_group = []
                current_tokens = 0
            current_group.append(seg)
            current_tokens += seg_tokens
        if current_group:
            result.append(current_group)
        return result

    def _merge_small_groups(self, groups: list[list[TranscriptSegment]]) -> list[list[TranscriptSegment]]:
        if not groups:
            return []
        result = []
        current_merged = list(groups[0])
        current_speaker = groups[0][0].speaker if groups[0] else None
        for group in groups[1:]:
            group_speaker = group[0].speaker if group else None
            group_text = " ".join(seg.text for seg in group)
            current_text = " ".join(seg.text for seg in current_merged)
            combined_tokens = estimate_tokens(current_text + " " + group_text)
            current_tokens = estimate_tokens(current_text)
            should_merge = (
                group_speaker == current_speaker
                and current_tokens < self.min_chunk_tokens
                and combined_tokens <= self.max_tokens
            )
            if should_merge:
                current_merged.extend(group)
            else:
                result.append(current_merged)
                current_merged = list(group)
                current_speaker = group_speaker
        if current_merged:
            result.append(current_merged)
        return result

    def _add_overlap_context(self, chunks: list[AudioChunk]) -> list[AudioChunk]:
        if len(chunks) <= 1:
            return chunks
        result = [chunks[0]]
        for i, chunk in enumerate(chunks[1:], 1):
            prev_chunk = chunks[i - 1]
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
