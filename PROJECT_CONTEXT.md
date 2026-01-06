# Audio RAG - Project Context

## Project Status: PRODUCTION READY ✅

**Last Updated:** January 2025

---

## What We Built

### Core System
| Component | Technology | Status |
|-----------|------------|--------|
| ASR | Faster-Whisper Large-v3 | ✅ Complete |
| Diarization | NVIDIA NeMo | ✅ Complete |
| Alignment | Custom word-to-speaker | ✅ Complete |
| Chunking | Speaker-turn (256 tokens) | ✅ Complete |
| Embeddings | BGE-M3 (dense + sparse) | ✅ Complete |
| Vector DB | Qdrant (hybrid search) | ✅ Complete |
| Reranking | BGE-reranker-base | ✅ Complete |
| Generation | Ollama (Llama 3.2) | ✅ Complete |

### Advanced Features
| Feature | Impact | Status |
|---------|--------|--------|
| Contextual Retrieval | +47% precision | ✅ Complete |
| Hybrid Search (RRF) | -31% latency | ✅ Complete |
| HyDE Query Expansion | +113% NLI | ✅ Complete |
| Real-time Streaming | 5-7s latency | ✅ Complete |

### Infrastructure
| Component | Technology | Status |
|-----------|------------|--------|
| API | FastAPI + Uvicorn | ✅ Complete |
| Frontend | React + Tailwind | ✅ Complete |
| Job Queue | Redis + RQ | ✅ Complete |
| Docker | Compose + GPU support | ✅ Complete |
| Kubernetes | Helm charts | ✅ Complete |

---

## Performance Metrics

### Retrieval Quality (CS229 Dataset)
| Config | Precision@5 | MRR | NDCG |
|--------|-------------|-----|------|
| Baseline | 0.425 | 0.650 | 0.652 |
| **Contextual** | **0.625** | **0.875** | **0.942** |

### Latency
| Operation | P50 |
|-----------|-----|
| Search only | 104ms |
| With reranking | 141ms |
| With generation | 584ms |
| Throughput | 7.1 qps |

### Streaming
| Metric | Value |
|--------|-------|
| Real-time factor | 0.66x |
| Chunk latency | 5-7 seconds |

---

## File Structure
```
audio-rag/
├── src/audio_rag/
│   ├── asr/
│   │   ├── whisper.py          # Batch ASR
│   │   └── streaming.py        # Real-time ASR
│   ├── diarization/            # NeMo speaker ID
│   ├── alignment/              # Word-to-speaker
│   ├── chunking/               # Speaker-turn chunking
│   ├── contextual/             # LLM context generation
│   ├── embeddings/             # BGE-M3 dense+sparse
│   ├── retrieval/              # Qdrant hybrid search
│   ├── reranking/              # BGE CrossEncoder
│   ├── expansion/              # HyDE
│   ├── generation/             # Ollama LLM
│   ├── evaluation/             # RAGAS metrics
│   ├── pipeline/               # Orchestration
│   ├── queue/                  # Redis jobs
│   ├── api/
│   │   └── v1/
│   │       ├── query.py        # Search endpoint
│   │       ├── ingest.py       # Upload endpoint
│   │       └── streaming.py    # WebSocket endpoint
│   └── config/                 # Pydantic schemas
├── frontend/
│   ├── src/
│   │   ├── pages/              # Home, Search, Upload, Streaming
│   │   ├── components/         # Layout, SearchBar, Results, etc.
│   │   └── api/                # Client SDK
│   └── Dockerfile              # Production build
├── k8s/
│   └── helm/audio-rag/
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/          # All K8s manifests
├── docs/
│   ├── COMPARISON.md           # Industry comparison
│   ├── INTERVIEW_GUIDE.md      # Technical deep dive
│   └── SALES_TECHNICAL_GUIDE.md # Sales documentation
├── docker-compose.yml          # Local deployment
├── Dockerfile.api              # API image
└── Dockerfile.worker           # GPU worker image
```

---

## Quick Commands

### Development
```bash
# Start services
docker compose up -d redis qdrant
ollama pull llama3.2:latest

# Run API
uvicorn audio_rag.api:create_app --factory --port 8000

# Run frontend
cd frontend && npm run dev
```

### Production (Docker)
```bash
docker compose --profile gpu up -d
```

### Production (Kubernetes)
```bash
helm install audio-rag ./k8s/helm/audio-rag -n audio-rag --create-namespace
```

### Testing
```bash
# Test streaming
uv run python scripts/test_streaming.py --file audio.wav

# Test WebSocket
uv run python scripts/test_streaming.py --websocket
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/query` | POST | Search with AI answers |
| `/api/v1/ingest` | POST | Upload audio file |
| `/api/v1/jobs/{id}` | GET | Check job status |
| `/api/v1/ws/transcribe` | WebSocket | Real-time streaming |
| `/api/v1/streaming/status` | GET | Streaming status |
| `/health/live` | GET | Liveness probe |
| `/health/ready` | GET | Readiness probe |

---

## Configuration

### Environment Variables
```bash
AUDIO_RAG_ENV=production
REDIS_URL=redis://localhost:6379/0
QDRANT_URL=http://localhost:6333
OLLAMA_URL=http://localhost:11434
```

### Key Config Options
```yaml
contextual:
  enabled: true           # +47% precision
  window_size: 1

retrieval:
  search_type: hybrid     # dense + sparse
  top_k: 5

reranking:
  backend: bge-reranker

expansion:
  backend: none           # or "hyde"
```

---

## Next Steps for UNT Pilot

1. [ ] Build Docker images and push to registry
2. [ ] Configure `values-unt.yaml` for UNT K8s
3. [ ] Ingest CS 5500 Fall 2025 lectures
4. [ ] Set up student access (API keys)
5. [ ] Monitor and tune performance
