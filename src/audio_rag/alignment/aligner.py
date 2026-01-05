"""ASR-Diarization alignment module.

Aligns word-level timestamps from ASR with speaker segments from diarization.

Research-backed implementation based on:
- Standard orchestration algorithm (arXiv 2409.00151)
- WhisperX alignment approach (m-bain/whisperX)
- DiarizationLM reconciliation (Interspeech 2024)
"""

from dataclasses import dataclass

from audio_rag.core import TranscriptSegment
from audio_rag.utils import get_logger

logger = get_logger(__name__)


@dataclass
class AlignedWord:
    """A word with speaker attribution."""

    word: str
    start: float
    end: float
    speaker: str | None


def align_words_to_speakers(
    words: list[dict],
    diarization_segments: list[TranscriptSegment],
    method: str = "overlap",
    tolerance: float = 0.5,
) -> list[AlignedWord]:
    """Align word-level timestamps with speaker diarization.

    Implements the standard orchestration algorithm from literature:
    1. If word overlaps with speaker segment -> assign speaker with max overlap
    2. If no overlap -> assign speaker with smallest temporal distance (within tolerance)

    Args:
        words: List of {"word": str, "start": float, "end": float}
        diarization_segments: Speaker segments from diarization
        method: "overlap" (default, research-backed) or "midpoint" (legacy)
        tolerance: Max gap (seconds) to search for nearest speaker when no overlap

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
        word_start = word_data["start"]
        word_end = word_data["end"]
        speaker = None

        if method == "overlap":
            # Step 1: Find speaker with maximum overlap (standard orchestration)
            max_overlap = 0.0
            for seg in diarization_segments:
                overlap_start = max(word_start, seg.start)
                overlap_end = min(word_end, seg.end)
                overlap = max(0.0, overlap_end - overlap_start)
                if overlap > max_overlap:
                    max_overlap = overlap
                    speaker = seg.speaker

            # Step 2: Fallback - find nearest segment if no overlap (within tolerance)
            if speaker is None and tolerance > 0:
                word_mid = (word_start + word_end) / 2
                min_distance = float("inf")
                for seg in diarization_segments:
                    # Calculate distance to segment
                    if word_mid < seg.start:
                        dist = seg.start - word_mid
                    elif word_mid > seg.end:
                        dist = word_mid - seg.end
                    else:
                        dist = 0  # Inside segment (shouldn't happen if overlap failed)
                    if dist < min_distance and dist <= tolerance:
                        min_distance = dist
                        speaker = seg.speaker
        else:
            # Midpoint method (legacy, less accurate with NeMo gaps)
            word_mid = (word_start + word_end) / 2
            for seg in diarization_segments:
                if seg.start <= word_mid <= seg.end:
                    speaker = seg.speaker
                    break

        aligned.append(
            AlignedWord(
                word=word_data["word"],
                start=word_start,
                end=word_end,
                speaker=speaker,
            )
        )

    # Log alignment stats
    assigned = sum(1 for w in aligned if w.speaker is not None)
    pct = (assigned / len(aligned) * 100) if aligned else 0
    logger.info(
        f"Aligned {assigned}/{len(aligned)} words ({pct:.1f}%) to speakers "
        f"[method={method}, tolerance={tolerance}s]"
    )

    return aligned


def build_speaker_transcript(
    aligned_words: list[AlignedWord],
    min_gap_seconds: float = 1.0,
    propagate_speaker: bool = True,
) -> list[TranscriptSegment]:
    """Build transcript segments from aligned words.

    Groups consecutive words by speaker, splitting on speaker changes
    or significant time gaps.

    Args:
        aligned_words: Words with speaker attribution
        min_gap_seconds: Gap threshold to force segment split
        propagate_speaker: If True, fill None speakers from neighbors

    Returns:
        List of TranscriptSegment with speaker attribution
    """
    if not aligned_words:
        return []

    # Step 1: Propagate speakers to fill None gaps (forward then backward fill)
    if propagate_speaker:
        aligned_words = _propagate_speakers(aligned_words)

    segments = []
    current_words: list[AlignedWord] = []
    current_speaker: str | None = None

    for word in aligned_words:
        # Check if we should start a new segment
        should_split = False

        if not current_words:
            # First word
            should_split = False
        elif word.speaker != current_speaker:
            # Speaker changed
            should_split = True
        elif word.start - current_words[-1].end > min_gap_seconds:
            # Time gap too large
            should_split = True

        if should_split and current_words:
            # Finalize current segment
            segments.append(_words_to_segment(current_words, current_speaker))
            current_words = []

        current_words.append(word)
        current_speaker = word.speaker

    # Don't forget the last segment
    if current_words:
        segments.append(_words_to_segment(current_words, current_speaker))

    logger.info(
        f"Built {len(segments)} transcript segments from {len(aligned_words)} words"
    )

    return segments


def _propagate_speakers(words: list[AlignedWord]) -> list[AlignedWord]:
    """Fill None speakers by propagating from neighbors.

    Strategy:
    1. Forward fill: If word has None speaker, use previous word's speaker
    2. Backward fill: If first words are None, use next non-None speaker
    """
    if not words:
        return words

    result = list(words)  # Copy to avoid mutating original

    # Forward fill
    last_speaker = None
    for i, word in enumerate(result):
        if word.speaker is not None:
            last_speaker = word.speaker
        elif last_speaker is not None:
            result[i] = AlignedWord(
                word=word.word,
                start=word.start,
                end=word.end,
                speaker=last_speaker,
            )

    # Backward fill (for words at the start with None)
    first_speaker = None
    for word in result:
        if word.speaker is not None:
            first_speaker = word.speaker
            break

    if first_speaker is not None:
        for i, word in enumerate(result):
            if word.speaker is None:
                result[i] = AlignedWord(
                    word=word.word,
                    start=word.start,
                    end=word.end,
                    speaker=first_speaker,
                )
            else:
                break  # Stop once we hit a non-None speaker

    # Log propagation stats
    original_none = sum(1 for w in words if w.speaker is None)
    final_none = sum(1 for w in result if w.speaker is None)
    if original_none > 0:
        logger.debug(
            f"Speaker propagation: {original_none} None -> {final_none} None "
            f"({original_none - final_none} filled)"
        )

    return result


def _words_to_segment(words: list[AlignedWord], speaker: str | None) -> TranscriptSegment:
    """Convert a list of words to a TranscriptSegment."""
    text = " ".join(w.word for w in words)
    return TranscriptSegment(
        text=text,
        start=words[0].start,
        end=words[-1].end,
        speaker=speaker,
        language="en",  # Default, could be passed in
    )
