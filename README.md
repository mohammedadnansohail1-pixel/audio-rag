# Audio RAG 🎙️

Production-grade Retrieval-Augmented Generation system for audio content. Ingest lectures, meetings, podcasts → Search and get AI-generated answers with speaker attribution.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

| Feature | Description |
|---------|-------------|
| **ASR** | Faster-Whisper large-v3 (99%+ accuracy, 100+ languages) |
| **Diarization** | NeMo speaker identification (2+ speakers) |
| **Hybrid Search** | Dense (BGE-M3) + Sparse (BM25) with RRF fusion |
| **Contextual Retrieval** | LLM-generated chunk context (+47% precision) |
| **Reranking** | BGE CrossEncoder for relevance scoring |
| **HyDE** | Query expansion via hypothetical documents |
| **Answer Generation** | Ollama LLM synthesis with citations |
| **Real-time Streaming** | WebSocket API for live transcription |
| **Web UI** | React dashboard for search, upload, streaming |
| **Multi-tenant** | Collection-based isolation per organization |
| **Kubernetes** | Helm charts for production deployment |

## 📊 Benchmarks

### Retrieval Quality

| Configuration | Precision@5 | MRR | NDCG |
|--------------|-------------|-----|------|
| Dense only | 0.425 | 0.650 | 0.652 |
| **Contextual + Hybrid** | **0.625** | **0.875** | **0.942** |

### Performance

| Metric | Value |
|--------|-------|
| Query latency | 141ms |
| Throughput | 7.1 qps |
| Streaming latency | 5-7 seconds |
| Real-time factor | 0.66x |

## 🚀 Quick Start

### Prerequisites
```bash
# Start infrastructure
docker run -d -p 6333:6333 qdrant/qdrant
docker run -d -p 6379:6379 redis
ollama pull llama3.2:latest
```

### Installation
```bash
git clone https://github.com/mohammedadnansohail1-pixel/audio-rag.git
cd audio-rag
uv sync

# Set CUDA path (Linux/WSL2)
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
```

### Usage
```python
from audio_rag import AudioRAG, AudioRAGConfig

# Initialize
config = AudioRAGConfig()
rag = AudioRAG(config)

# Ingest (with contextual for best quality)
result = rag.ingest('lecture.wav', enable_contextual=True)
print(f"Indexed {result.num_chunks} chunks")

# Search
result = rag.query(
    'What is gradient descent?',
    search_type='hybrid',
    enable_reranking=True,
    generate_answer=True,
)
print(result.generated_answer)
```

### Web UI
```bash
cd frontend && npm install && npm run dev
# Open http://localhost:3000
```

### API
```bash
uvicorn audio_rag.api:create_app --factory --port 8000

# Query
curl -X POST http://localhost:8000/api/v1/query \
  -H "X-API-Key: dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is ML?", "generate_answer": true}'
```

### Real-time Streaming
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/transcribe');
ws.onmessage = (e) => console.log(JSON.parse(e.data).text);
// Send audio chunks from microphone...
```

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────────────────┐
│                         INGESTION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│  Audio → Whisper → NeMo → Align → Chunk → Context → BGE-M3 → Qdrant│
│           ASR     Diarize  Speaker  256tok   LLM    Dense+Sparse    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                          QUERY PIPELINE                             │
├─────────────────────────────────────────────────────────────────────┤
│  Query → (HyDE) → BGE-M3 → Hybrid Search → Rerank → Ollama → Answer │
└─────────────────────────────────────────────────────────────────────┘
```

## 📦 Deployment

### Docker Compose
```bash
docker compose --profile gpu up -d
```

### Kubernetes
```bash
helm install audio-rag ./k8s/helm/audio-rag -n audio-rag --create-namespace
```

See [k8s/README.md](k8s/README.md) for production deployment guide.

## 📁 Project Structure
```
audio-rag/
├── src/audio_rag/        # Core Python package
│   ├── asr/              # Whisper + streaming
│   ├── diarization/      # NeMo speaker ID
│   ├── contextual/       # LLM context generation
│   ├── retrieval/        # Qdrant hybrid search
│   ├── reranking/        # BGE CrossEncoder
│   ├── generation/       # Ollama answers
│   ├── evaluation/       # RAGAS metrics
│   ├── pipeline/         # Orchestration
│   ├── api/              # FastAPI endpoints
│   └── queue/            # Redis job queue
├── frontend/             # React web UI
├── k8s/helm/             # Kubernetes Helm charts
├── docs/                 # Documentation
└── tests/                # Unit + integration tests
```

## 📚 Documentation

- [Technical Interview Guide](docs/INTERVIEW_GUIDE.md) - Deep dive into all decisions
- [Industry Comparison](docs/COMPARISON.md) - How we compare to AssemblyAI, Deepgram, Glean
- [Sales & Technical Guide](docs/SALES_TECHNICAL_GUIDE.md) - Complete buyer documentation
- [Kubernetes Guide](k8s/README.md) - Production deployment

## 🔧 Configuration
```yaml
# configs/base.yaml
asr:
  backend: faster-whisper
  model_size: large-v3

contextual:
  enabled: true
  window_size: 1

retrieval:
  search_type: hybrid
  top_k: 5

reranking:
  backend: bge-reranker

generation:
  backend: ollama
  model: llama3.2:latest
```

## 📈 Evaluation
```python
from audio_rag.evaluation import RAGEvaluator, CS229_EVAL_DATASET

evaluator = RAGEvaluator(use_nli=True)
results = evaluator.evaluate_dataset(pipeline, CS229_EVAL_DATASET)
evaluator.print_summary(results)
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- [BGE-M3](https://github.com/FlagOpen/FlagEmbedding)
- [Qdrant](https://qdrant.tech/)
- [Ollama](https://ollama.ai/)
- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
