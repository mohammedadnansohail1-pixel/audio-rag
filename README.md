# Audio RAG

Production-grade, config-driven Audio RAG system with multilingual support.

## Features

- **Config-Driven**: YAML configuration with environment overrides
- **Registry Pattern**: Pluggable backends for all components
- **Multi-Component Pipeline**:
  - ASR: Faster-Whisper (multilingual, GPU accelerated)
  - Diarization: PyAnnote (speaker identification)
  - Embeddings: BGE-M3 (multilingual, 1024 dim)
  - Retrieval: Qdrant (hybrid search)
  - TTS: Edge-TTS (100+ languages)
- **Resource Management**: VRAM tracking, model lazy-loading

## Quick Start
```bash
# Install dependencies
uv sync

# Set CUDA library path (WSL2)
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"

# Ingest audio
uv run python scripts/run.py ingest audio.mp3 --no-diarization

# Query
uv run python scripts/run.py query "What was discussed?"

# Query with audio response
uv run python scripts/run.py query "Summarize the main points" --audio

# Check status
uv run python scripts/run.py status
```

## Configuration
```yaml
# configs/base.yaml
asr:
  backend: "faster-whisper"
  model_size: "large-v3"  # tiny, base, small, medium, large-v2, large-v3
  device: "auto"          # cuda, cpu, auto

embedding:
  backend: "bge-m3"
  model: "BAAI/bge-m3"

retrieval:
  backend: "qdrant"
  search_type: "hybrid"   # dense, sparse, hybrid

tts:
  backend: "edge-tts"
```

## Usage as Library
```python
from audio_rag import AudioRAG

# Initialize
rag = AudioRAG.from_config(env="development")

# Ingest
result = rag.ingest("podcast.mp3", enable_diarization=True)
print(f"Ingested {result.num_chunks} chunks")

# Query
response = rag.query("What did they discuss about AI?")
for r in response.results:
    print(f"[{r.chunk.speaker}] {r.chunk.text}")

# Query with audio
response = rag.query("Summarize", generate_audio=True, audio_output_path="out.mp3")
```

## Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                        AudioRAG                                  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Ingest    │    │    Query    │    │  Resource   │         │
│  │  Pipeline   │    │  Pipeline   │    │  Manager    │         │
│  └──────┬──────┘    └──────┬──────┘    └─────────────┘         │
│         │                  │                                    │
│  ┌──────▼──────┐    ┌──────▼──────┐                            │
│  │ ASR         │    │ Embedder    │◄──── Shared                │
│  │ Diarizer    │    │ Retriever   │◄──── Components            │
│  │ Aligner     │    │ TTS         │                            │
│  │ Chunker     │    └─────────────┘                            │
│  └─────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure
```
audio-rag/
├── configs/
│   ├── base.yaml           # Base configuration
│   ├── development.yaml    # Dev overrides
│   └── production.yaml     # Production settings
├── src/audio_rag/
│   ├── config/             # Pydantic schemas, loader
│   ├── core/               # Registry, base classes, exceptions
│   ├── asr/                # Faster-Whisper
│   ├── diarization/        # PyAnnote
│   ├── alignment/          # Transcript-speaker alignment
│   ├── chunking/           # Speaker-turn, fixed chunking
│   ├── embeddings/         # BGE-M3
│   ├── retrieval/          # Qdrant
│   ├── tts/                # Edge-TTS, Piper
│   ├── pipeline/           # Ingestion, Query, Orchestrator
│   ├── resources/          # VRAM/memory management
│   └── utils/              # Logging, decorators
├── scripts/
│   └── run.py              # CLI
└── tests/
```

## Requirements

- Python 3.11
- CUDA 12.x (optional, for GPU)
- ~16GB RAM recommended
- ~8GB VRAM for GPU mode

## Diarization Setup

PyAnnote requires HuggingFace authentication:

1. Accept license at https://huggingface.co/pyannote/speaker-diarization-3.1
2. Create token at https://huggingface.co/settings/tokens
3. Set environment variable: `export HF_TOKEN="your_token"`

## License

MIT
