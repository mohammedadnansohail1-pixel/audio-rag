#!/usr/bin/env python3
"""Test client for real-time streaming ASR.

Usage:
    # Test with microphone (requires sounddevice)
    uv run python scripts/test_streaming.py --mic
    
    # Test with audio file
    uv run python scripts/test_streaming.py --file audio.wav
    
    # Test WebSocket connection
    uv run python scripts/test_streaming.py --websocket
"""

import argparse
import asyncio
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np


def test_with_file(audio_path: str, chunk_seconds: float = 5.0):
    """Test streaming ASR with an audio file."""
    from audio_rag.asr import StreamingASR, StreamingConfig
    from audio_rag.config import ASRConfig
    
    print(f"Testing streaming ASR with: {audio_path}")
    
    # Load audio file
    path = Path(audio_path)
    if not path.exists():
        print(f"Error: File not found: {audio_path}")
        return
    
    # Read audio
    if path.suffix == '.wav':
        with wave.open(str(path), 'rb') as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            n_frames = wf.getnframes()
            audio_bytes = wf.readframes(n_frames)
            
            # Convert to float32
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Convert to mono if stereo
            if n_channels == 2:
                audio = audio.reshape(-1, 2).mean(axis=1)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                print(f"Warning: Resampling from {sample_rate}Hz to 16000Hz")
                from scipy import signal
                audio = signal.resample(audio, int(len(audio) * 16000 / sample_rate))
                sample_rate = 16000
    else:
        # Use librosa for other formats
        try:
            import librosa
            audio, sample_rate = librosa.load(str(path), sr=16000, mono=True)
        except ImportError:
            print("Error: Install librosa for non-WAV files: pip install librosa")
            return
    
    print(f"Audio loaded: {len(audio)/16000:.1f}s, {sample_rate}Hz")
    
    # Create streaming ASR
    asr_config = ASRConfig(
        backend="faster-whisper",
        model_size="large-v3",
        device="auto",
        compute_type="float16",
    )
    
    streaming_config = StreamingConfig(
        chunk_duration=chunk_seconds,
        overlap_duration=1.0,
    )
    
    streamer = StreamingASR(asr_config, streaming_config)
    
    # Simulate streaming
    chunk_samples = int(chunk_seconds * 16000)
    
    def audio_generator():
        """Generate audio chunks."""
        for i in range(0, len(audio), chunk_samples // 10):  # Send in smaller pieces
            chunk = audio[i:i + chunk_samples // 10]
            if len(chunk) > 0:
                yield chunk
            time.sleep(0.05)  # Simulate real-time
    
    print("\nStreaming transcription:")
    print("-" * 50)
    
    total_start = time.time()
    
    for result in streamer.process_stream_sync(audio_generator()):
        if result.text:
            print(f"[{result.start:.1f}s - {result.end:.1f}s] ({result.processing_time_ms:.0f}ms)")
            print(f"  {result.text}")
            print()
    
    total_time = time.time() - total_start
    audio_duration = len(audio) / 16000
    
    print("-" * 50)
    print(f"Audio duration: {audio_duration:.1f}s")
    print(f"Processing time: {total_time:.1f}s")
    print(f"Real-time factor: {total_time/audio_duration:.2f}x")


async def test_websocket(host: str = "localhost", port: int = 8000):
    """Test WebSocket streaming endpoint."""
    try:
        import websockets
    except ImportError:
        print("Error: Install websockets: pip install websockets")
        return
    
    uri = f"ws://{host}:{port}/api/v1/ws/transcribe"
    print(f"Connecting to: {uri}")
    
    try:
        async with websockets.connect(uri) as ws:
            print("Connected!")
            
            # Receive ready message
            msg = await ws.recv()
            print(f"Server: {msg}")
            
            # Send some test audio (silence)
            print("\nSending 5 seconds of silence...")
            
            # 5 seconds of silence at 16kHz, int16
            silence = np.zeros(16000 * 5, dtype=np.int16)
            await ws.send(silence.tobytes())
            
            # Wait for response
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                print(f"Server: {msg}")
            except asyncio.TimeoutError:
                print("Timeout waiting for response")
            
            # Send stop command
            await ws.send(json.dumps({"command": "stop"}))
            
            # Get final messages
            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    print(f"Server: {msg}")
            except asyncio.TimeoutError:
                pass
            
            print("\nWebSocket test complete!")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure the API server is running:")
        print("  uvicorn audio_rag.api:create_app --factory --port 8000")


def test_with_microphone(duration: float = 30.0):
    """Test streaming ASR with microphone input."""
    try:
        import sounddevice as sd
    except ImportError:
        print("Error: Install sounddevice: pip install sounddevice")
        print("  uv add sounddevice")
        return
    
    from audio_rag.asr import StreamingASR, StreamingConfig
    from audio_rag.config import ASRConfig
    
    print("Testing streaming ASR with microphone")
    print(f"Recording for {duration} seconds...")
    print("Speak into your microphone!")
    print("-" * 50)
    
    # Create streaming ASR
    asr_config = ASRConfig(
        backend="faster-whisper",
        model_size="large-v3",
        device="auto",
        compute_type="float16",
    )
    
    streaming_config = StreamingConfig(
        chunk_duration=5.0,
        overlap_duration=1.0,
    )
    
    streamer = StreamingASR(asr_config, streaming_config)
    
    # Audio queue
    import queue
    audio_queue = queue.Queue()
    
    def audio_callback(indata, frames, time_info, status):
        """Callback for sounddevice."""
        if status:
            print(f"Audio status: {status}")
        audio_queue.put(indata.copy())
    
    def audio_generator():
        """Generate audio from queue."""
        start_time = time.time()
        while time.time() - start_time < duration:
            try:
                chunk = audio_queue.get(timeout=0.5)
                yield chunk.flatten()
            except queue.Empty:
                continue
    
    # Start recording
    with sd.InputStream(
        samplerate=16000,
        channels=1,
        dtype=np.float32,
        callback=audio_callback,
        blocksize=1600,  # 100ms chunks
    ):
        print("\n🎤 Recording... (speak now)\n")
        
        for result in streamer.process_stream_sync(audio_generator()):
            if result.text:
                print(f"[{result.start:.1f}s] {result.text}")
    
    print("\n" + "-" * 50)
    print("Recording complete!")


def main():
    parser = argparse.ArgumentParser(description="Test streaming ASR")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mic", action="store_true", help="Test with microphone")
    group.add_argument("--file", type=str, help="Test with audio file")
    group.add_argument("--websocket", action="store_true", help="Test WebSocket endpoint")
    
    parser.add_argument("--duration", type=float, default=30.0, help="Recording duration (mic mode)")
    parser.add_argument("--chunk", type=float, default=5.0, help="Chunk duration in seconds")
    parser.add_argument("--host", type=str, default="localhost", help="WebSocket host")
    parser.add_argument("--port", type=int, default=8000, help="WebSocket port")
    
    args = parser.parse_args()
    
    if args.mic:
        test_with_microphone(args.duration)
    elif args.file:
        test_with_file(args.file, args.chunk)
    elif args.websocket:
        asyncio.run(test_websocket(args.host, args.port))


if __name__ == "__main__":
    main()
