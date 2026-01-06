"""Streaming ASR implementation for real-time transcription."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Callable, Iterator
import numpy as np

from audio_rag.asr.base import ASRRegistry
from audio_rag.core import TranscriptSegment, Word, ASRError
from audio_rag.config import ASRConfig
from audio_rag.utils import get_logger

logger = get_logger(__name__)


class StreamingState(Enum):
    """State of the streaming session."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    STOPPED = "stopped"


@dataclass
class StreamingConfig:
    """Configuration for streaming ASR."""
    sample_rate: int = 16000
    chunk_duration: float = 5.0
    overlap_duration: float = 1.0
    min_chunk_duration: float = 1.0
    vad_filter: bool = True
    language: str | None = None


@dataclass
class StreamingResult:
    """Result from streaming transcription."""
    text: str
    start: float
    end: float
    is_final: bool
    words: list[Word] = field(default_factory=list)
    language: str | None = None
    processing_time_ms: float = 0.0


class AudioBuffer:
    """Thread-safe audio buffer for accumulating chunks."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._buffer: np.ndarray = np.array([], dtype=np.float32)
        self._lock = asyncio.Lock()
        self._total_duration: float = 0.0
    
    async def add(self, audio: np.ndarray) -> None:
        async with self._lock:
            self._buffer = np.append(self._buffer, audio.astype(np.float32))
    
    async def get_and_trim(self, keep_seconds: float = 1.0) -> np.ndarray:
        async with self._lock:
            audio = self._buffer.copy()
            keep_samples = int(keep_seconds * self.sample_rate)
            self._buffer = self._buffer[-keep_samples:] if len(self._buffer) > keep_samples else np.array([], dtype=np.float32)
            self._total_duration += len(audio) / self.sample_rate - keep_seconds
            return audio
    
    async def get_duration(self) -> float:
        async with self._lock:
            return len(self._buffer) / self.sample_rate
    
    async def clear(self) -> None:
        async with self._lock:
            self._buffer = np.array([], dtype=np.float32)
            self._total_duration = 0.0
    
    @property
    def total_duration(self) -> float:
        return self._total_duration


class StreamingASR:
    """Real-time streaming ASR using chunked Whisper processing."""
    
    def __init__(
        self, 
        asr_config: ASRConfig,
        streaming_config: StreamingConfig | None = None,
    ):
        self.asr_config = asr_config
        self.streaming_config = streaming_config or StreamingConfig()
        self._model = None
        self._state = StreamingState.IDLE
        self._buffer = AudioBuffer(self.streaming_config.sample_rate)
        self._session_start: float = 0.0
        self._chunk_index: int = 0
        
        logger.info(
            f"StreamingASR initialized: chunk={self.streaming_config.chunk_duration}s, "
            f"overlap={self.streaming_config.overlap_duration}s"
        )
    
    @property
    def state(self) -> StreamingState:
        return self._state
    
    @property
    def is_active(self) -> bool:
        return self._state in (StreamingState.LISTENING, StreamingState.PROCESSING)
    
    def _load_model(self) -> None:
        if self._model is not None:
            return
        
        try:
            from faster_whisper import WhisperModel
            
            device = self._resolve_device(self.asr_config.device)
            logger.info(f"Loading Whisper {self.asr_config.model_size} for streaming...")
            
            self._model = WhisperModel(
                self.asr_config.model_size,
                device=device,
                compute_type=self.asr_config.compute_type,
            )
            logger.info("Whisper model loaded for streaming")
            
        except Exception as e:
            raise ASRError(f"Failed to load Whisper model: {e}")
    
    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    
    async def reset(self) -> None:
        """Reset state for new session (keeps model loaded)."""
        await self._buffer.clear()
        self._session_start = 0.0
        self._chunk_index = 0
        self._state = StreamingState.IDLE
        logger.debug("StreamingASR reset")
    
    async def start(self) -> None:
        """Start streaming session."""
        # Allow restart from any state
        if self._state not in (StreamingState.IDLE, StreamingState.STOPPED):
            if self._state == StreamingState.LISTENING:
                logger.debug("Already listening, continuing session")
                return
            raise ASRError(f"Cannot start: state is {self._state}")
        
        self._load_model()
        await self._buffer.clear()
        self._session_start = time.time()
        self._chunk_index = 0
        self._state = StreamingState.LISTENING
        
        logger.info("Streaming session started")
    
    async def stop(self) -> StreamingResult | None:
        """Stop streaming session and process remaining audio."""
        if self._state == StreamingState.STOPPED:
            return None
        
        if self._state == StreamingState.IDLE:
            self._state = StreamingState.STOPPED
            return None
        
        self._state = StreamingState.PROCESSING
        
        remaining_duration = await self._buffer.get_duration()
        result = None
        
        if remaining_duration >= self.streaming_config.min_chunk_duration:
            audio = await self._buffer.get_and_trim(keep_seconds=0)
            result = self._transcribe_chunk(audio, is_final=True)
        
        await self._buffer.clear()
        self._state = StreamingState.STOPPED
        
        logger.info(
            f"Streaming session stopped: {self._chunk_index} chunks, "
            f"{self._buffer.total_duration:.1f}s total"
        )
        
        return result
    
    async def add_audio(self, audio: bytes | np.ndarray) -> StreamingResult | None:
        """Add audio chunk and return result if ready."""
        # Auto-start if needed
        if self._state in (StreamingState.IDLE, StreamingState.STOPPED):
            await self.start()
        
        if self._state == StreamingState.PROCESSING:
            # Wait briefly for processing to complete
            await asyncio.sleep(0.1)
        
        if not self.is_active:
            raise ASRError(f"Cannot add audio: state is {self._state}")
        
        if isinstance(audio, bytes):
            audio = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        
        await self._buffer.add(audio)
        
        duration = await self._buffer.get_duration()
        
        if duration >= self.streaming_config.chunk_duration:
            self._state = StreamingState.PROCESSING
            
            audio_data = await self._buffer.get_and_trim(
                keep_seconds=self.streaming_config.overlap_duration
            )
            result = self._transcribe_chunk(audio_data, is_final=False)
            
            self._state = StreamingState.LISTENING
            return result
        
        return None
    
    async def process_stream(
        self, 
        audio_stream: AsyncIterator[bytes | np.ndarray],
    ) -> AsyncIterator[StreamingResult]:
        """Process an async stream of audio chunks."""
        await self.start()
        
        try:
            async for chunk in audio_stream:
                result = await self.add_audio(chunk)
                if result:
                    yield result
            
            final_result = await self.stop()
            if final_result:
                yield final_result
                
        except Exception as e:
            logger.error(f"Stream processing error: {e}")
            self._state = StreamingState.STOPPED
            raise
    
    def process_stream_sync(
        self,
        audio_stream: Iterator[bytes | np.ndarray],
    ) -> Iterator[StreamingResult]:
        """Process a synchronous stream of audio chunks."""
        self._load_model()
        self._session_start = time.time()
        self._chunk_index = 0
        
        buffer = np.array([], dtype=np.float32)
        chunk_samples = int(self.streaming_config.chunk_duration * self.streaming_config.sample_rate)
        overlap_samples = int(self.streaming_config.overlap_duration * self.streaming_config.sample_rate)
        
        for chunk in audio_stream:
            if isinstance(chunk, bytes):
                chunk = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            
            buffer = np.append(buffer, chunk.astype(np.float32))
            
            if len(buffer) >= chunk_samples:
                result = self._transcribe_chunk(buffer, is_final=False)
                yield result
                buffer = buffer[-overlap_samples:]
        
        if len(buffer) >= int(self.streaming_config.min_chunk_duration * self.streaming_config.sample_rate):
            result = self._transcribe_chunk(buffer, is_final=True)
            yield result
    
    def _transcribe_chunk(
        self, 
        audio: np.ndarray, 
        is_final: bool = False,
    ) -> StreamingResult:
        """Transcribe a single audio chunk."""
        start_time = time.time()
        
        chunk_start = self._buffer.total_duration
        chunk_duration = len(audio) / self.streaming_config.sample_rate
        
        try:
            segments, info = self._model.transcribe(
                audio,
                language=self.streaming_config.language,
                vad_filter=self.streaming_config.vad_filter,
                word_timestamps=True,
            )
            
            texts = []
            words = []
            
            for segment in segments:
                texts.append(segment.text.strip())
                
                if segment.words:
                    for w in segment.words:
                        words.append(Word(
                            word=w.word,
                            start=chunk_start + w.start,
                            end=chunk_start + w.end,
                            confidence=w.probability if hasattr(w, 'probability') else None,
                        ))
            
            text = " ".join(texts).strip()
            processing_time = (time.time() - start_time) * 1000
            
            self._chunk_index += 1
            
            result = StreamingResult(
                text=text,
                start=chunk_start,
                end=chunk_start + chunk_duration,
                is_final=is_final,
                words=words,
                language=info.language,
                processing_time_ms=processing_time,
            )
            
            logger.debug(
                f"Chunk {self._chunk_index}: {len(text)} chars, "
                f"{processing_time:.0f}ms, final={is_final}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return StreamingResult(
                text="",
                start=chunk_start,
                end=chunk_start + chunk_duration,
                is_final=is_final,
                processing_time_ms=(time.time() - start_time) * 1000,
            )


@ASRRegistry.register("streaming-whisper")
class StreamingWhisperASR(StreamingASR):
    """Registered streaming ASR backend."""
    pass
