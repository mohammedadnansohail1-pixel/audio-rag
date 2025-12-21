"""Transcript-diarization alignment module."""

from audio_rag.alignment.aligner import (
    AlignedWord,
    align_words_to_speakers,
    align_segments_to_speakers,
    build_speaker_transcript,
)

__all__ = [
    "AlignedWord",
    "align_words_to_speakers",
    "align_segments_to_speakers",
    "build_speaker_transcript",
]
