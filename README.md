# Audio RAG 🎙️

Production-grade Retrieval-Augmented Generation system for audio content. Ingest lectures, meetings, podcasts → Search and get AI-generated answers with speaker attribution.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

| Feature | Description |
|---------|-------------|
| **ASR** | Faster-Whisper large-v3 (99%+ accuracy, 100+ languages) |
| **Diarization** | NeMo speaker identification (2+ speakers) |
| **Hybrid Search** | Dense (BGE-M3) + Sparse (BM25) with RRF fusion |
| **Contextual Retrieval** | LLM-generated chunk context (+47% precision) |
| **Reranking** | BGE CrossEncoder for relevance scoring |
| **HyDE** | Query expansion via hypothetical documents |
| **Answer Generation** | Ollama LLM synthesis with citations |
| **Multi-tenant** | Collection-based isolation per organization |
| **Production API** | FastAPI with rate limiting, auth, Redis queue |

## 📊 Benchmarks

### Retrieval Quality (CS229 Lecture Dataset)

| Configuration | Precision@K | MRR | NDCG | Hit Rate | Latency |
|--------------|-------------|-----|------|----------|---------|
| Dense only | 0.425 | 0.650 | 0.652 | 0.750 | 2956ms |
| Hybrid (Dense + BM25) | 0.425 | 0.650 | 0.652 | 0.750 | 1824ms |
| **Contextual + Hybrid** | **0.625** | **0.875** | **0.942** | **0.875** | **2876ms** |
| Contextual + HyDE | 0.675 | 0.875 | 0.990 | 0.875 | 4718ms |

### Production Performance

| Metric | Value |
|--------|-------|
| Query latency (warm) | 141ms |
| Throughput | 7.1 queries/sec |
| Concurrent users | 22+ (100% success) |
| Model load time | 4.8s (one-time) |

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/audio-rag.git
cd audio-rag

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .

# Set CUDA path (WSL2/Linux)
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
```

### Prerequisites
```bash
# Start Qdrant (vector store)
docker run -d -p 6333:6333 qdrant/qdrant

# Start Redis (job queue)
docker run -d -p 6379:6379 redis

# Pull Ollama model
ollama pull llama3.2:latest
```

### Basic Usage
```python
from audio_rag.pipeline import AudioRAG
from audio_rag.config import AudioRAGConfig

# Initialize
config = AudioRAGConfig()
config.retrieval.collection_name = 'my_lectures'
rag = AudioRAG(config)

# Ingest audio (with contextual for best quality)
result = rag.ingest('lecture.wav', enable_contextual=True)
print(f"Ingested {result.num_chunks} chunks, {len(result.speakers)} speakers")

# Query
result = rag.query(
    'What is gradient descent?',
    search_type='hybrid',
    enable_reranking=True,
    generate_answer=True,
)
print(result.generated_answer)
```

### CLI Usage
```bash
# Ingest
uv run python -m audio_rag.cli ingest lecture.mp3 --collection cs229

# Query
uv run python -m audio_rag.cli query "What is RAG?" --collection cs229

# Status
uv run python -m audio_rag.cli status
```

### API Usage
```bash
# Start server
uvicorn audio_rag.api:create_app --factory --port 8000

# Query endpoint
curl -X POST http://localhost:8000/api/v1/query \
  -H "X-API-Key: dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "collection_name": "cs229",
    "search_type": "hybrid",
    "enable_reranking": true,
    "generate_answer": true
  }'
```

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────────────────┐
│                         INGESTION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│  Audio → Whisper → NeMo → Alignment → Chunking → Context → Embed   │
│    │       ASR    Diarize   Speaker     Split      LLM     BGE-M3  │
│    │                        Labels      256tok    Context   Dense   │
│    │                                              (Ollama)  +Sparse │
│    └────────────────────────────────────────────────────────────────┤
│                                                              Qdrant │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                          QUERY PIPELINE                             │
├─────────────────────────────────────────────────────────────────────┤
│  Query → (HyDE) → Embed → Hybrid Search → Rerank → Generate        │
│    │     Expand   BGE-M3   Dense+BM25     BGE-CE   Ollama          │
│    │     Ollama           RRF Fusion                                │
└─────────────────────────────────────────────────────────────────────┘
```

## ⚙️ Configuration
```yaml
# configs/base.yaml
asr:
  backend: "faster-whisper"
  model_size: "large-v3"
  device: "auto"

diarization:
  backend: "nemo"
  device: "auto"

chunking:
  strategy: "speaker_turn"
  max_tokens: 256
  min_chunk_tokens: 30

contextual:
  enabled: true           # Add LLM context to chunks
  window_size: 1          # Surrounding chunks for context

embedding:
  backend: "bge-m3"
  model: "BAAI/bge-m3"
  use_sparse: true        # Enable BM25 sparse vectors

retrieval:
  backend: "qdrant"
  search_type: "hybrid"   # dense, sparse, or hybrid
  top_k: 5

reranking:
  backend: "bge-reranker"
  model: "BAAI/bge-reranker-base"
  initial_k: 20           # Retrieve 20, rerank to top_k

expansion:
  backend: "none"         # Set to "hyde" to enable

generation:
  backend: "ollama"
  model: "llama3.2:latest"
```

## 📁 Project Structure
```
audio-rag/
├── src/audio_rag/
│   ├── asr/              # Speech recognition (Whisper)
│   ├── diarization/      # Speaker identification (NeMo)
│   ├── alignment/        # Word-to-speaker alignment
│   ├── chunking/         # Text chunking strategies
│   ├── contextual/       # LLM context generation
│   ├── embeddings/       # BGE-M3 dense+sparse
│   ├── retrieval/        # Qdrant hybrid search
│   ├── reranking/        # BGE CrossEncoder
│   ├── expansion/        # HyDE query expansion
│   ├── generation/       # Ollama answer synthesis
│   ├── evaluation/       # RAGAS, NLI metrics
│   ├── pipeline/         # Orchestration
│   ├── queue/            # Redis job queue
│   ├── api/              # FastAPI endpoints
│   └── config/           # Pydantic schemas
├── configs/              # YAML configurations
├── tests/                # Unit & integration tests
└── docs/                 # Documentation
```

## 🔬 Evaluation
```python
from audio_rag.evaluation import RAGEvaluator, CS229_EVAL_DATASET

evaluator = RAGEvaluator(use_nli=True, use_semantic=True)
results = evaluator.evaluate_dataset(pipeline, CS229_EVAL_DATASET)
summary = evaluator.summarize_results(results)
evaluator.print_summary(summary)
```

### Available Metrics

| Category | Metrics |
|----------|---------|
| Retrieval | Precision@K, Recall@K, MRR, NDCG, Hit Rate |
| Generation | Faithfulness, Answer Similarity, NLI Score, BLEU |
| Performance | Latency (avg, p95), Throughput |

## 🏢 Multi-Tenant Support
```python
# Each organization gets isolated collection
rag.ingest('lecture1.wav', collection_name='university_a')
rag.ingest('meeting.wav', collection_name='enterprise_b')

# Queries are isolated
result = rag.query('budget discussion', collection_name='enterprise_b')
```

## 🔧 Advanced Features

### HyDE Query Expansion
```python
# Generates hypothetical answer, embeds that instead of raw query
result = rag.query('What is attention?', enable_hyde=True)
print(f"HyDE used: {result.hyde_used}")
print(f"Expanded: {result.expanded_query[:100]}...")
```

### Contextual Retrieval
```python
# Add context during ingestion (requires re-indexing)
result = rag.ingest('lecture.wav', enable_contextual=True)
# Chunks now have: "[Context: This discusses...]\nOriginal text..."
```

### Custom Evaluation Dataset
```python
from audio_rag.evaluation import EvalDataset, EvalSample

dataset = EvalDataset(
    name="my_eval",
    samples=[
        EvalSample(
            question="What is X?",
            ground_truth="X is...",
            ground_truth_contexts=["keyword1", "keyword2"],
        ),
    ],
)
dataset.to_json("my_eval.json")
```

## 📈 Comparison with Industry Solutions

See [docs/COMPARISON.md](docs/COMPARISON.md) for detailed comparison.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) - ASR
- [NeMo](https://github.com/NVIDIA/NeMo) - Diarization
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) - BGE-M3
- [Qdrant](https://qdrant.tech/) - Vector database
- [Ollama](https://ollama.ai/) - Local LLM inference
