"""WebSocket endpoint for real-time audio transcription."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from pydantic import BaseModel

from audio_rag.asr import StreamingASR, StreamingConfig, StreamingResult
from audio_rag.config import ASRConfig
from audio_rag.utils import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["streaming"])

# Shared model instance (singleton for efficiency)
_streaming_asr: StreamingASR | None = None
_asr_lock = asyncio.Lock()


async def get_streaming_asr() -> StreamingASR:
    """Get or create streaming ASR instance."""
    global _streaming_asr
    
    async with _asr_lock:
        if _streaming_asr is None:
            asr_config = ASRConfig(
                backend="faster-whisper",
                model_size="large-v3",
                device="auto",
                compute_type="float16",
            )
            _streaming_asr = StreamingASR(
                asr_config=asr_config,
                streaming_config=StreamingConfig(),
            )
        return _streaming_asr


class TranscriptMessage(BaseModel):
    """WebSocket message for transcript results."""
    type: str = "transcript"
    text: str
    start: float
    end: float
    is_final: bool
    language: str | None = None
    processing_time_ms: float = 0.0
    words: list[dict] | None = None


class ErrorMessage(BaseModel):
    """WebSocket message for errors."""
    type: str = "error"
    message: str
    code: str | None = None


class StatusMessage(BaseModel):
    """WebSocket message for status updates."""
    type: str = "status"
    state: str
    message: str


def result_to_message(result: StreamingResult) -> dict[str, Any]:
    """Convert StreamingResult to WebSocket message."""
    return TranscriptMessage(
        text=result.text,
        start=result.start,
        end=result.end,
        is_final=result.is_final,
        language=result.language,
        processing_time_ms=result.processing_time_ms,
        words=[
            {"word": w.word, "start": w.start, "end": w.end}
            for w in result.words
        ] if result.words else None,
    ).model_dump()


@router.websocket("/ws/transcribe")
async def websocket_transcribe(
    websocket: WebSocket,
    language: str | None = Query(None, description="Language code (e.g., 'en')"),
    chunk_duration: float = Query(5.0, ge=1.0, le=30.0, description="Seconds per chunk"),
):
    """Real-time audio transcription via WebSocket."""
    await websocket.accept()
    
    client_id = id(websocket)
    logger.info(f"WebSocket connected: {client_id}")
    
    # Get shared ASR instance and reset for new session
    try:
        asr = await get_streaming_asr()
        # Update config for this session
        asr.streaming_config.language = language
        asr.streaming_config.chunk_duration = chunk_duration
        # Reset state for new connection
        await asr.reset()
    except Exception as e:
        logger.error(f"Failed to initialize ASR: {e}")
        await websocket.send_json(
            ErrorMessage(message=str(e), code="ASR_INIT_FAILED").model_dump()
        )
        await websocket.close()
        return
    
    session_active = False
    
    try:
        await websocket.send_json(
            StatusMessage(state="ready", message="Send audio data to begin").model_dump()
        )
        
        while True:
            message = await websocket.receive()
            
            if message["type"] == "websocket.disconnect":
                break
            
            # Handle text commands
            if "text" in message:
                try:
                    cmd = json.loads(message["text"])
                    
                    if cmd.get("command") == "stop":
                        logger.info(f"Stop command received: {client_id}")
                        
                        if session_active:
                            final_result = await asr.stop()
                            if final_result and final_result.text:
                                await websocket.send_json(result_to_message(final_result))
                            session_active = False
                        
                        await websocket.send_json(
                            StatusMessage(state="stopped", message="Session stopped").model_dump()
                        )
                        break
                    
                    elif cmd.get("command") == "reset":
                        logger.info(f"Reset command received: {client_id}")
                        await asr.reset()
                        session_active = False
                        await websocket.send_json(
                            StatusMessage(state="ready", message="Session reset").model_dump()
                        )
                    
                except json.JSONDecodeError:
                    await websocket.send_json(
                        ErrorMessage(message="Invalid JSON command", code="INVALID_COMMAND").model_dump()
                    )
                continue
            
            # Handle binary audio data
            if "bytes" in message:
                audio_data = message["bytes"]
                
                if not session_active:
                    await asr.start()
                    session_active = True
                    await websocket.send_json(
                        StatusMessage(state="listening", message="Processing audio").model_dump()
                    )
                
                try:
                    result = await asr.add_audio(audio_data)
                    
                    if result and result.text:
                        await websocket.send_json(result_to_message(result))
                        
                except Exception as e:
                    logger.error(f"Audio processing error: {e}")
                    await websocket.send_json(
                        ErrorMessage(message=str(e), code="PROCESSING_ERROR").model_dump()
                    )
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {client_id}")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json(
                ErrorMessage(message=str(e), code="SERVER_ERROR").model_dump()
            )
        except:
            pass
    
    finally:
        if session_active:
            try:
                await asr.stop()
            except:
                pass
        # Reset for next connection
        try:
            await asr.reset()
        except:
            pass
        
        logger.info(f"WebSocket session ended: {client_id}")


@router.get("/streaming/status")
async def get_streaming_status():
    """Get streaming ASR status."""
    global _streaming_asr
    
    if _streaming_asr is None:
        return {
            "initialized": False,
            "message": "Streaming ASR not initialized. Connect to /ws/transcribe to initialize.",
        }
    
    return {
        "initialized": True,
        "state": _streaming_asr.state.value,
        "config": {
            "sample_rate": _streaming_asr.streaming_config.sample_rate,
            "chunk_duration": _streaming_asr.streaming_config.chunk_duration,
            "overlap_duration": _streaming_asr.streaming_config.overlap_duration,
        },
    }
