"""Transcript-diarization alignment algorithms."""

from dataclasses import dataclass
from audio_rag.core import TranscriptSegment, AlignmentError
from audio_rag.utils import get_logger, timed

logger = get_logger(__name__)


@dataclass
class AlignedWord:
    """A word with timing and speaker attribution."""
    word: str
    start: float
    end: float
    speaker: str | None = None


def align_words_to_speakers(
    words: list[dict],
    diarization_segments: list[TranscriptSegment],
) -> list[AlignedWord]:
    """Align word-level timestamps with speaker diarization.
    
    Uses midpoint alignment: assigns each word to the speaker
    active at the word's midpoint.
    
    Args:
        words: List of {"word": str, "start": float, "end": float}
        diarization_segments: Speaker segments from diarization
        
    Returns:
        List of AlignedWord with speaker attribution
    """
    if not words:
        return []
    
    if not diarization_segments:
        logger.warning("No diarization segments, words will have no speaker")
        return [
            AlignedWord(word=w["word"], start=w["start"], end=w["end"], speaker=None)
            for w in words
        ]
    
    aligned = []
    
    for word_data in words:
        word_mid = (word_data["start"] + word_data["end"]) / 2
        speaker = None
        
        # Find speaker active at word midpoint
        for seg in diarization_segments:
            if seg.start <= word_mid <= seg.end:
                speaker = seg.speaker
                break
        
        aligned.append(
            AlignedWord(
                word=word_data["word"],
                start=word_data["start"],
                end=word_data["end"],
                speaker=speaker,
            )
        )
    
    # Log alignment stats
    assigned = sum(1 for w in aligned if w.speaker is not None)
    logger.info(f"Aligned {assigned}/{len(aligned)} words to speakers")
    
    return aligned


def align_segments_to_speakers(
    transcript_segments: list[TranscriptSegment],
    diarization_segments: list[TranscriptSegment],
    method: str = "overlap",
) -> list[TranscriptSegment]:
    """Align transcript segments with speaker diarization.
    
    Args:
        transcript_segments: Segments from ASR (with text, no speaker)
        diarization_segments: Segments from diarization (with speaker, no text)
        method: Alignment method - 'overlap' (max overlap) or 'midpoint'
        
    Returns:
        Transcript segments with speaker attribution
    """
    if not transcript_segments:
        return []
    
    if not diarization_segments:
        logger.warning("No diarization segments, transcripts will have no speaker")
        return transcript_segments
    
    aligned = []
    
    for seg in transcript_segments:
        if method == "midpoint":
            speaker = _find_speaker_at_midpoint(seg, diarization_segments)
        else:  # overlap
            speaker = _find_speaker_by_overlap(seg, diarization_segments)
        
        aligned.append(
            TranscriptSegment(
                text=seg.text,
                start=seg.start,
                end=seg.end,
                speaker=speaker,
                confidence=seg.confidence,
                language=seg.language,
            )
        )
    
    # Log alignment stats
    assigned = sum(1 for s in aligned if s.speaker is not None)
    logger.info(f"Aligned {assigned}/{len(aligned)} segments to speakers")
    
    return aligned


def _find_speaker_at_midpoint(
    segment: TranscriptSegment,
    diarization_segments: list[TranscriptSegment],
) -> str | None:
    """Find speaker active at segment midpoint."""
    midpoint = (segment.start + segment.end) / 2
    
    for diar_seg in diarization_segments:
        if diar_seg.start <= midpoint <= diar_seg.end:
            return diar_seg.speaker
    
    return None


def _find_speaker_by_overlap(
    segment: TranscriptSegment,
    diarization_segments: list[TranscriptSegment],
) -> str | None:
    """Find speaker with maximum overlap with segment."""
    max_overlap = 0.0
    best_speaker = None
    
    for diar_seg in diarization_segments:
        # Calculate overlap
        overlap_start = max(segment.start, diar_seg.start)
        overlap_end = min(segment.end, diar_seg.end)
        overlap = max(0.0, overlap_end - overlap_start)
        
        if overlap > max_overlap:
            max_overlap = overlap
            best_speaker = diar_seg.speaker
    
    return best_speaker


@timed
def build_speaker_transcript(
    words: list[AlignedWord],
    min_gap_seconds: float = 1.0,
) -> list[TranscriptSegment]:
    """Build transcript segments grouped by speaker turns.
    
    Groups consecutive words by speaker, creating new segments
    when speaker changes or there's a significant gap.
    
    Args:
        words: Aligned words with speaker attribution
        min_gap_seconds: Gap threshold to force new segment
        
    Returns:
        List of transcript segments grouped by speaker turn
    """
    if not words:
        return []
    
    segments = []
    current_speaker = words[0].speaker
    current_words = [words[0].word]
    current_start = words[0].start
    current_end = words[0].end
    
    for i, word in enumerate(words[1:], 1):
        prev_word = words[i - 1]
        gap = word.start - prev_word.end
        speaker_changed = word.speaker != current_speaker
        large_gap = gap > min_gap_seconds
        
        # Start new segment if speaker changes or large gap
        if speaker_changed or large_gap:
            # Save current segment
            segments.append(
                TranscriptSegment(
                    text=" ".join(current_words),
                    start=current_start,
                    end=current_end,
                    speaker=current_speaker,
                )
            )
            
            # Start new segment
            current_speaker = word.speaker
            current_words = [word.word]
            current_start = word.start
            current_end = word.end
        else:
            # Continue current segment
            current_words.append(word.word)
            current_end = word.end
    
    # Don't forget last segment
    if current_words:
        segments.append(
            TranscriptSegment(
                text=" ".join(current_words),
                start=current_start,
                end=current_end,
                speaker=current_speaker,
            )
        )
    
    logger.info(f"Built {len(segments)} speaker turns from {len(words)} words")
    return segments
