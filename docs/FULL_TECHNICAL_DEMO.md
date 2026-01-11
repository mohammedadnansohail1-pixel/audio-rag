# Audio RAG: Full Technical Demo Script

**Duration:** 20-30 minutes
**Target:** Technical audience, investors, hiring managers

---

## Demo Structure

| Part | Duration | Content |
|------|----------|---------|
| 1. Problem & Solution | 2 min | Why we built this |
| 2. Architecture Overview | 3 min | System diagram, components |
| 3. Ingestion Pipeline | 8 min | Live step-by-step processing |
| 4. Query Pipeline | 5 min | Search, reranking, generation |
| 5. Advanced Features | 4 min | Contextual, HyDE, benchmarks |
| 6. Real-time Streaming | 3 min | WebSocket demo |
| 7. Production Deployment | 3 min | Docker, K8s, scaling |
| 8. Wrap-up | 2 min | Summary, questions |

---

# PART 1: Problem & Solution (2 min)

## Script

> "Universities record thousands of hours of lectures every semester. Students can't search them. They rewatch entire recordings to find one concept.
>
> Audio RAG solves this. It's a production-grade system that:
> 1. Transcribes audio with speaker identification
> 2. Creates a searchable knowledge base
> 3. Generates AI answers with citations
>
> Everything runs on-premise. Zero API costs. Your data never leaves your servers.
>
> Let me show you how it works - from the ground up."

---

# PART 2: Architecture Overview (3 min)

## Show Project Structure
```bash
# Show the codebase structure
tree ~/projects/audio-rag/src/audio_rag -L 1
```

**Output to show:**
```
audio_rag/
├── asr/           # Whisper transcription
├── diarization/   # NeMo speaker ID
├── alignment/     # Word-to-speaker mapping
├── chunking/      # Text segmentation
├── contextual/    # LLM context generation
├── embeddings/    # BGE-M3 vectors
├── retrieval/     # Qdrant search
├── reranking/     # CrossEncoder
├── expansion/     # HyDE
├── generation/    # Ollama answers
├── evaluation/    # RAGAS metrics
├── pipeline/      # Orchestration
├── api/           # FastAPI
└── queue/         # Redis jobs
```

## Draw Architecture (on whiteboard or show diagram)
```
┌─────────────────────────────────────────────────────────────────────┐
│                       INGESTION PIPELINE                            │
│                                                                     │
│  Audio ──▶ Whisper ──▶ NeMo ──▶ Align ──▶ Chunk ──▶ Context ──▶ Embed│
│            ASR       Diarize   Words    256tok    LLM      BGE-M3  │
│                                to Spk                      Dense+  │
│                                                            Sparse  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                            ┌──────────┐
                            │  Qdrant  │
                            │  Vector  │
                            │    DB    │
                            └──────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        QUERY PIPELINE                               │
│                                                                     │
│  Query ──▶ (HyDE) ──▶ Embed ──▶ Hybrid ──▶ Rerank ──▶ Ollama ──▶ Answer│
│            Expand     BGE-M3   Search    Top-5     Generate        │
└─────────────────────────────────────────────────────────────────────┘
```

## Script

> "The system has two main pipelines:
>
> **Ingestion** processes audio files:
> - Whisper transcribes with word-level timestamps
> - NeMo identifies different speakers
> - We align words to speakers
> - Chunk by speaker turns (256 tokens)
> - Generate contextual descriptions with LLM
> - Create dense + sparse embeddings
> - Store in Qdrant
>
> **Query** handles search:
> - Optional HyDE expansion for better matching
> - Hybrid search: semantic + keyword
> - Rerank with CrossEncoder
> - Generate answer with Ollama
>
> Let me show you each step live."

---

# PART 3: Ingestion Pipeline - Live Demo (8 min)

## Setup
```bash
# Start Python REPL
cd ~/projects/audio-rag
uv run python
```

## Step 3.1: ASR (Whisper)
```python
# Load the ASR module
from audio_rag.asr import FasterWhisperASR
from audio_rag.config import ASRConfig

# Configure Whisper
config = ASRConfig(model_size="large-v3", device="cuda")
asr = FasterWhisperASR(config)
asr.load()

# Transcribe
segments = asr.transcribe("data/samples/test_audio.mp3")

# Show results
for seg in segments[:3]:
    print(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")
    if seg.words:
        print(f"  Words: {len(seg.words)} with timestamps")
```

**Script:**
> "This is Whisper Large-v3 running locally on our GPU. It gives us:
> - Word-level timestamps
> - 4.2% word error rate - best in class
> - 100+ language support
>
> Notice each word has a timestamp. We'll use this for speaker alignment."

## Step 3.2: Diarization (NeMo)
```python
# Load diarization
from audio_rag.diarization import NeMoDiarizer
from audio_rag.config import DiarizationConfig

diarizer = NeMoDiarizer(DiarizationConfig(device="cuda"))
diarizer.load()

# Run diarization
speaker_turns = diarizer.diarize("data/samples/test_audio.mp3")

for turn in speaker_turns[:5]:
    print(f"[{turn.start:.1f}s - {turn.end:.1f}s] {turn.speaker}")
```

**Script:**
> "NeMo from NVIDIA identifies WHO spoke WHEN. 
> This is clustering-based - it groups similar voice embeddings.
> Now we know 'Speaker 0 talked from 0-15 seconds'."

## Step 3.3: Alignment
```python
# Align words to speakers
from audio_rag.alignment import align_transcript_to_speakers

aligned_segments = align_transcript_to_speakers(segments, speaker_turns)

for seg in aligned_segments[:3]:
    print(f"[{seg.speaker}] {seg.text[:50]}...")
```

**Script:**
> "Now we know not just WHAT was said, but WHO said it.
> This is crucial for queries like 'What did the professor say about X?'"

## Step 3.4: Chunking
```python
# Create chunks
from audio_rag.chunking import SpeakerTurnChunker
from audio_rag.config import ChunkingConfig

chunker = SpeakerTurnChunker(ChunkingConfig(
    max_tokens=256,
    overlap_tokens=50,
    min_chunk_tokens=30
))

chunks = chunker.chunk(aligned_segments)

print(f"Created {len(chunks)} chunks")
for chunk in chunks[:2]:
    print(f"\n[{chunk.speaker}] ({chunk.start:.1f}s - {chunk.end:.1f}s)")
    print(f"  {chunk.text[:100]}...")
```

**Script:**
> "We chunk by speaker turns with 256 token max.
> The 50-token overlap ensures we don't lose context at boundaries.
> Each chunk preserves speaker attribution."

## Step 3.5: Contextual Processing (The Secret Sauce)
```python
# Add context to chunks
from audio_rag.contextual import ContextualProcessor
from audio_rag.config import ContextualConfig

processor = ContextualProcessor(ContextualConfig(window_size=1))

# Process one chunk to show
contextualized = processor.generate_context(chunks[0], chunks[:3])
print("BEFORE:")
print(f"  {chunks[0].text[:100]}...")
print("\nAFTER (with context):")
print(f"  {contextualized.text[:200]}...")
```

**Script:**
> "This is Anthropic's Contextual Retrieval technique - published October 2024.
>
> Standard RAG loses context when you chunk. 'The gradient is computed...' - what gradient?
>
> We use an LLM to prepend context: 'This section from a machine learning lecture 
> discusses backpropagation. The gradient is computed...'
>
> This improved our precision by 47%. Huge win."

## Step 3.6: Embedding
```python
# Create embeddings
from audio_rag.embeddings import BGEM3Embedder
from audio_rag.config import EmbeddingConfig

embedder = BGEM3Embedder(EmbeddingConfig(use_sparse=True))
embedder.load()

# Embed chunks
results = embedder.embed([c.text for c in chunks[:3]])

print(f"Dense vector: {len(results[0].dense)} dimensions")
print(f"Sparse vector: {len(results[0].sparse.indices)} non-zero entries")
```

**Script:**
> "BGE-M3 creates BOTH dense and sparse vectors in one pass.
> - Dense: 1024 dimensions for semantic meaning
> - Sparse: Like learned BM25 for keyword matching
>
> We search both and combine with Reciprocal Rank Fusion."

## Step 3.7: Storage (Qdrant)
```python
# Store in Qdrant
from audio_rag.retrieval import QdrantRetriever
from audio_rag.config import RetrievalConfig

retriever = QdrantRetriever(RetrievalConfig(
    collection_name="demo_collection",
    search_type="hybrid"
))
retriever.initialize()

# Add chunks
retriever.add_chunks(chunks, results)
print(f"Stored {len(chunks)} chunks in Qdrant")
```

**Script:**
> "Qdrant handles hybrid search natively. 
> It stores both vector types and does parallel retrieval."

---

# PART 4: Query Pipeline - Live Demo (5 min)

## Step 4.1: Basic Search
```python
# Search
query = "What is the weather forecast?"
query_embedding = embedder.embed([query])[0]

results = retriever.search(
    query_embedding, 
    top_k=5,
    search_type="hybrid"
)

for r in results:
    print(f"[Score: {r.score:.3f}] [{r.chunk.speaker}]")
    print(f"  {r.chunk.text[:80]}...")
```

**Script:**
> "Hybrid search combines:
> - Dense: semantic similarity (weather ≈ forecast, temperature)
> - Sparse: keyword matching (exact terms)
> - RRF fusion combines by rank, not score"

## Step 4.2: Reranking
```python
# Rerank results
from audio_rag.reranking import BGEReranker
from audio_rag.config import RerankingConfig

reranker = BGEReranker(RerankingConfig())
reranker.load()

reranked = reranker.rerank(query, [r.chunk for r in results], top_k=3)

print("After reranking:")
for chunk, score in reranked:
    print(f"[Score: {score:.3f}] {chunk.text[:80]}...")
```

**Script:**
> "The CrossEncoder looks at query AND document together.
> More accurate than bi-encoder, but slower.
> That's why we only rerank top-20 → top-5."

## Step 4.3: Answer Generation
```python
# Generate answer
from audio_rag.generation import OllamaGenerator
from audio_rag.config import GenerationConfig

generator = OllamaGenerator(GenerationConfig(model="llama3.2:latest"))

context = "\n\n".join([f"[{c.speaker}]: {c.text}" for c, _ in reranked])
answer = generator.generate(query, context)

print("Question:", query)
print("\nAnswer:", answer)
```

**Script:**
> "Ollama runs Llama 3.2 locally. No API costs.
> The prompt includes speaker attribution so answers can say 
> 'According to Professor Smith...'"

---

# PART 5: Advanced Features (4 min)

## 5.1: Benchmark Results
```python
# Show evaluation results
print("""
RETRIEVAL QUALITY (CS229 Dataset)
═══════════════════════════════════════════════════
Configuration        Precision@5   MRR     NDCG
───────────────────────────────────────────────────
Dense only           0.425        0.650   0.652
Hybrid (Dense+BM25)  0.425        0.650   0.652
Contextual           0.625        0.875   0.942
Contextual + HyDE    0.675        0.875   0.990
═══════════════════════════════════════════════════

IMPROVEMENTS:
- Contextual Retrieval: +47% precision
- HyDE Query Expansion: +113% NLI score
- Hybrid Search: -31% latency

PERFORMANCE:
- Query latency: 141ms
- Throughput: 7.1 queries/sec
- Streaming: 0.66x real-time
""")
```

**Script:**
> "We built a full evaluation framework using RAGAS metrics.
> The numbers show contextual retrieval is the biggest win.
> This isn't just theory - these are measured improvements on real lecture data."

## 5.2: HyDE Demonstration
```python
# Show HyDE expansion
from audio_rag.expansion import HyDEExpander
from audio_rag.config import ExpansionConfig

expander = HyDEExpander(ExpansionConfig())

query = "What is attention?"
hypothetical = expander.expand(query)

print("Original query:", query)
print("\nHyDE expansion (hypothetical answer):")
print(hypothetical[:300] + "...")
```

**Script:**
> "HyDE - Hypothetical Document Embeddings.
> A 3-word query embeds very differently than a 200-word chunk.
> We generate a hypothetical answer and embed THAT instead.
> Bridges the query-document gap. +113% NLI improvement."

---

# PART 6: Real-time Streaming (3 min)

## Terminal Demo
```python
# Streaming demo
from audio_rag.asr import StreamingASR, StreamingConfig
from audio_rag.config import ASRConfig
import librosa

# Load audio
audio, sr = librosa.load("data/samples/test_audio.mp3", sr=16000, mono=True)
print(f"Loaded {len(audio)/16000:.1f}s of audio")

# Create streamer
asr_config = ASRConfig(model_size="large-v3", device="cuda")
streaming_config = StreamingConfig(chunk_duration=5.0)
streamer = StreamingASR(asr_config, streaming_config)

# Simulate streaming
def audio_generator():
    chunk_size = 16000  # 1 second chunks
    for i in range(0, len(audio), chunk_size):
        yield audio[i:i+chunk_size]

print("\nStreaming transcription:")
for result in streamer.process_stream_sync(audio_generator()):
    if result.text:
        print(f"[{result.start:.1f}s] {result.text}")
```

**Script:**
> "Real-time streaming processes audio as it arrives.
> 5-second chunks, 1-second overlap for context.
> 0.66x real-time factor - faster than real-time.
> Perfect for live lectures, meetings, or accessibility."

## Show WebSocket Code
```bash
# Show the WebSocket endpoint
cat ~/projects/audio-rag/src/audio_rag/api/v1/streaming.py | head -50
```

---

# PART 7: Production Deployment (3 min)

## Docker
```bash
# Show Docker setup
cat ~/projects/audio-rag/docker-compose.yml
```
```bash
# Start with Docker
docker compose --profile gpu up -d
```

**Script:**
> "Docker Compose for single-server deployment.
> Redis for job queues, Qdrant for vectors, GPU worker for ML."

## Kubernetes
```bash
# Show Helm chart
ls ~/projects/audio-rag/k8s/helm/audio-rag/templates/

# Show values
head -50 ~/projects/audio-rag/k8s/helm/audio-rag/values.yaml
```
```bash
# Deploy
helm install audio-rag ./k8s/helm/audio-rag -n audio-rag --create-namespace
```

**Script:**
> "For enterprise, we have full Helm charts:
> - API with HorizontalPodAutoscaler (2-10 replicas)
> - GPU workers with NVIDIA tolerations
> - Qdrant StatefulSet with persistent volumes
> - Ingress with TLS
>
> Production-grade orchestration."

## Multi-tenant
```python
# Show multi-tenant isolation
print("""
MULTI-TENANT ARCHITECTURE
═════════════════════════════════════════════

Tenant A (CS Department)
├── Collection: cs229
├── Collection: cs231
└── API Key: tenant-a-key

Tenant B (Business School)
├── Collection: mba101
└── API Key: tenant-b-key

- Complete data isolation
- Per-tenant rate limiting
- Separate API keys
""")
```

---

# PART 8: Wrap-up (2 min)

## Summary
```python
print("""
AUDIO RAG - SUMMARY
════════════════════════════════════════════════════════

WHAT WE BUILT:
- End-to-end audio search with AI answers
- 7+ ML models coordinated
- Real-time streaming
- Production Kubernetes deployment

KEY INNOVATIONS:
- Contextual Retrieval: +47% precision
- Hybrid Search: Dense + Sparse + RRF
- Speaker Attribution: "Professor said..."
- 100% on-premise: Zero API costs

TECH STACK:
- Whisper Large-v3 (ASR)
- NVIDIA NeMo (Diarization)
- BGE-M3 (Embeddings)
- Qdrant (Vector DB)
- Ollama/Llama (Generation)
- FastAPI + React (API + UI)
- Kubernetes/Helm (Deployment)

PERFORMANCE:
- 141ms query latency
- 7.1 queries/sec
- 0.66x real-time streaming

PROJECT LEVEL: Staff Engineer (L6-L7)
════════════════════════════════════════════════════════
""")
```

**Script:**
> "That's Audio RAG - from audio file to AI answers in milliseconds.
>
> Key takeaways:
> 1. Contextual Retrieval is a game-changer - 47% better precision
> 2. Hybrid search gives you semantic AND keyword matching
> 3. Everything runs locally - complete data privacy
> 4. Production-ready with Kubernetes Helm charts
>
> This is the kind of system companies raise millions to build.
> I built it as a graduate student.
>
> Questions?"

---

# APPENDIX: Quick Commands

## Start Demo Environment
```bash
# Infrastructure
docker start qdrant redis
ollama serve &

# API (for full demo)
cd ~/projects/audio-rag
uv run uvicorn audio_rag.api:create_app --factory --port 8000 &

# Interactive Python (for pipeline demo)
cd ~/projects/audio-rag
uv run python
```

## Full Pipeline in One Script
```python
# Complete ingestion + query demo
from audio_rag import AudioRAG, AudioRAGConfig

# Initialize
config = AudioRAGConfig()
config.retrieval.collection_name = "demo"
rag = AudioRAG(config)

# Ingest
result = rag.ingest("data/samples/test_audio.mp3", enable_contextual=True)
print(f"Ingested: {result.num_chunks} chunks, {len(result.speakers)} speakers")

# Query
result = rag.query(
    "What is the weather forecast?",
    search_type="hybrid",
    enable_reranking=True,
    generate_answer=True,
)
print(f"Answer: {result.generated_answer}")
```

## Recording Tips

1. **Terminal Setup:**
   - Dark theme (easier to read)
   - Large font (16-18pt)
   - Tmux/split screen for multiple views

2. **Code Flow:**
   - Have all commands ready in a script
   - Copy-paste, don't type live
   - Add pauses for explanation

3. **Visual Aids:**
   - Architecture diagram (draw or show)
   - Performance numbers on screen
   - Highlight key metrics

4. **Pacing:**
   - Slower for complex concepts
   - Faster for boilerplate
   - Pause after key insights
