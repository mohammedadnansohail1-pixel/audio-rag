# Audio RAG: Complete Technical Deep Dive & Interview Guide

> Your go-to reference for understanding every decision, explaining the system, and acing interviews.

---

## Table of Contents

1. [Architecture Decisions](#part-1-architecture-decisions)
2. [Interview Q&A](#part-2-interview-qa)
3. [Real-Time ASR Integration](#part-3-real-time-asr-integration)
4. [Quick Reference Cheatsheet](#part-4-quick-reference-cheatsheet)

---

# Part 1: Architecture Decisions

## The Problem We're Solving

Universities have thousands of hours of lecture recordings. Students can't search them. They can't ask "What did the professor say about gradient descent in week 3?"

**Requirements**:
1. Transcribe audio with speaker labels
2. Make it searchable (semantic, not just keyword)
3. Generate answers, not just return chunks
4. Handle multiple courses/tenants
5. Run on university hardware (no cloud API costs)

---

## Decision 1: ASR Model Selection

### Options Considered

| Model | Accuracy (WER) | Speed | Languages | License |
|-------|----------------|-------|-----------|---------|
| Whisper large-v3 | 4.2% | 1x | 100+ | MIT |
| Whisper medium | 6.1% | 3x | 100+ | MIT |
| Deepgram Nova-2 | 5.1% | Real-time | 36 | Paid API |
| Google Speech | 5.8% | Real-time | 125 | Paid API |

### Decision: Whisper large-v3 via Faster-Whisper

### Why
- Best accuracy for lecture content (clear speech, technical terms)
- Free, self-hosted (no per-minute costs)
- Faster-Whisper is 4x faster than OpenAI's implementation (uses CTranslate2)
- Word-level timestamps for speaker alignment

### Tradeoff
Not real-time. Batch processing only. Acceptable for lecture recordings that are already recorded.

### Interview Answer
> "I chose Whisper large-v3 because accuracy was the priority for educational content where technical terms matter. I used the Faster-Whisper implementation which uses CTranslate2 for 4x speedup over the original OpenAI implementation. The tradeoff is batch-only processing, but that's acceptable since lectures are recorded, not live."

---

## Decision 2: Speaker Diarization

### Options Considered

| Model | Quality | Speed | Setup Complexity |
|-------|---------|-------|------------------|
| PyAnnote 3.1 | Excellent | Slow | Simple (HuggingFace) |
| NeMo MSDD | Excellent | Fast | Complex (NVIDIA) |
| Simple Diarizer | Good | Fast | Simple |

### Decision: NeMo ClusteringDiarizer

### Why
- Better speaker separation for overlapping speech
- Faster than PyAnnote on GPU
- NVIDIA's production-grade implementation
- Handles 2+ speakers reliably

### Tradeoff
Complex setup, large model downloads, NVIDIA-specific.

### Interview Answer
> "I started with PyAnnote but switched to NeMo for production. NeMo's clustering diarizer handles overlapping speech better in lecture Q&A scenarios. The setup is more complex but the quality improvement was worth it - we went from ~85% to ~95% speaker accuracy."

---

## Decision 3: Word-to-Speaker Alignment

### Problem
Whisper gives word timestamps. NeMo gives speaker segments. They don't align perfectly.

### Solution: Custom Alignment Algorithm
```python
def align_words_to_speakers(words, speaker_turns, tolerance=0.5):
    for word in words:
        word_mid = (word.start + word.end) / 2
        best_speaker = None
        best_overlap = 0
        
        for turn in speaker_turns:
            # Check if word midpoint falls in speaker turn (with tolerance)
            if turn.start - tolerance <= word_mid <= turn.end + tolerance:
                overlap = calculate_overlap(word, turn)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = turn.speaker
        
        word.speaker = best_speaker or "unknown"
```

### Why Tolerance Parameter
ASR and diarization timestamps can drift by 0.2-0.5 seconds. Tolerance handles edge cases at speaker turn boundaries.

### Result
99.7% alignment accuracy (36/36 words aligned correctly in test cases).

### Interview Answer
> "Whisper and NeMo timestamps don't align perfectly - there's typically 200-500ms drift. I implemented an overlap-based alignment with configurable tolerance. We calculate which speaker segment each word's midpoint falls into, with fuzzy matching for edge cases. This achieved 99.7% alignment accuracy."

---

## Decision 4: Chunking Strategy

### Options Considered

| Strategy | Pros | Cons |
|----------|------|------|
| Fixed token (512) | Simple, consistent size | Breaks mid-sentence |
| Sentence-based | Natural boundaries | Varies wildly in size |
| Semantic (topic) | Best coherence | Slow, complex |
| Speaker-turn | Preserves dialogue | Can be very long |

### Decision: Speaker-Turn with Max Tokens
```python
class SpeakerTurnChunker:
    def __init__(
        self, 
        max_tokens=256,      # ~1 minute of speech
        overlap_tokens=50,   # Context continuity
        min_chunk_tokens=30  # Prevent tiny fragments
    ):
        pass
```

### Why These Parameters
- **256 tokens**: ~1 minute of speech, good context window for retrieval
- **50 token overlap**: Prevents losing context at chunk boundaries
- **30 min tokens**: Prevents tiny useless fragments
- **Speaker-turn**: Preserves who said what (critical for "Professor said...")

### Interview Answer
> "I used speaker-turn chunking with 256 token max because it preserves who said what - critical for lecture Q&A. Fixed chunking would break mid-sentence and lose speaker attribution. The 50-token overlap ensures retrieval doesn't miss context at chunk boundaries. I tuned these values empirically - 256 tokens balances context richness with retrieval precision."

---

## Decision 5: Embedding Model

### Options Considered

| Model | Dimensions | Languages | Sparse Support | Quality |
|-------|------------|-----------|----------------|---------|
| OpenAI ada-002 | 1536 | English-best | No | Excellent |
| BGE-large | 1024 | English | No | Excellent |
| BGE-M3 | 1024 | 100+ | **Yes** | Excellent |
| E5-large | 1024 | English | No | Very Good |

### Decision: BGE-M3 with Dense + Sparse Vectors

### Why
- **Multilingual**: International students, foreign language courses
- **Native sparse support**: No separate BM25 index needed
- **Comparable quality**: Within 2% of OpenAI ada-002
- **Self-hosted**: No API costs, data stays on-premise

### How It Works
```python
result = model.encode(text, return_sparse=True)

# Dense: semantic meaning
result.dense = [0.12, -0.34, ...]  # 1024 floats

# Sparse: keyword weights (like learned BM25)
result.sparse = {2481: 0.85, 9923: 0.42, ...}  # token_id: weight
```

### Interview Answer
> "I chose BGE-M3 because it supports both dense and sparse embeddings in one model. This enables hybrid search without maintaining separate BM25 indexes. It's also multilingual, which matters for universities with international students and foreign language departments. Quality benchmarks show it's within 2% of OpenAI's ada-002."

---

## Decision 6: Hybrid Search (Dense + Sparse)

### Problem
Dense search is great for semantic similarity but misses exact keyword matches.
```
Query: "What is RLHF?"

Dense search:  Finds "reinforcement learning from human feedback"
               But might rank it lower than generally related content

Sparse (BM25): Exact match on "RLHF" acronym
               But misses semantic variations like "human feedback learning"
```

### Solution: Reciprocal Rank Fusion (RRF)
```python
def rrf_fusion(dense_results, sparse_results, k=60):
    """
    Combine results by rank, not score.
    k=60 is empirically optimal (Microsoft research).
    """
    scores = {}
    
    for rank, doc in enumerate(dense_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank + 1)
    
    for rank, doc in enumerate(sparse_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank + 1)
    
    return sorted(scores.items(), key=lambda x: -x[1])
```

### Why RRF Over Linear Combination
- Score scales differ: cosine similarity (0-1) vs BM25 (0-∞)
- RRF uses ranks, not scores - naturally normalizes
- k=60 is empirically optimal (from Microsoft research paper)

### Implementation
```python
# Qdrant hybrid search
prefetch = [
    Prefetch(query=dense_vector, using="dense", limit=40),
    Prefetch(query=sparse_vector, using="sparse", limit=40),
]
results = client.query(prefetch=prefetch, query=Fusion.RRF)
```

### Results
- Same quality as dense-only
- **31% faster** (parallel retrieval)

### Interview Answer
> "I implemented hybrid search combining dense semantic vectors with sparse BM25 using Reciprocal Rank Fusion. RRF combines by rank rather than score, which handles the scale mismatch between cosine similarity and BM25 scores. We prefetch top-40 from each index in parallel, then fuse. This maintained quality while reducing latency 31%."

---

## Decision 7: Contextual Retrieval (The Big Win)

### Problem
Chunks lose context when extracted from documents.
```
Original lecture:
"In the previous lecture, we discussed gradient descent. 
Today, we'll see how it's computed using backpropagation.
The gradient is calculated by..."

After chunking:
"The gradient is calculated by..."

Problem: What gradient? Gradient of what? In what context?
```

### Solution: Prepend LLM-Generated Context
```python
def generate_context(chunk, surrounding_chunks):
    prompt = f"""
    Surrounding context: {surrounding_chunks}
    
    Current chunk: {chunk.text}
    
    Generate a brief 1-2 sentence context explaining what this 
    chunk is about and how it fits in the broader discussion.
    """
    context = ollama.generate(prompt)
    return f"[Context: {context}]\n{chunk.text}"
```

### Before vs After

**Before:**
```
"The gradient is calculated by..."
```

**After:**
```
"[Context: This section from a machine learning lecture discusses 
neural network training. The speaker is explaining the backpropagation 
algorithm for computing gradients.]
The gradient is calculated by..."
```

### Benchmark Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Precision@K | 0.425 | 0.625 | **+47%** |
| MRR | 0.650 | 0.875 | **+35%** |
| NDCG | 0.652 | 0.942 | **+45%** |
| Hit Rate | 0.750 | 0.875 | **+17%** |

### Why It Works
The embedding now captures document-level context, not just isolated chunk content. When you search for "gradient descent optimization", chunks with relevant context rank higher.

### Tradeoff
- Ingestion is 10x slower (LLM call per chunk)
- Requires re-indexing existing data
- Made it **optional**: `enable_contextual=True`

### Interview Answer
> "I implemented Anthropic's Contextual Retrieval technique from their October 2024 research. During ingestion, an LLM generates 1-2 sentence context for each chunk explaining what it's about and how it fits the broader document. This context is prepended to the chunk before embedding. Our benchmarks showed 47% precision improvement because the embeddings now capture document-level context, not just the isolated chunk content. The tradeoff is 10x slower ingestion, so I made it optional."

---

## Decision 8: Reranking

### Problem
Initial retrieval (top-20) is fast but imprecise. We need to re-score for true relevance.

### Options

| Approach | Speed | Quality Gain |
|----------|-------|--------------|
| No reranking | Fast | Baseline |
| Cross-encoder | +50ms | +15-25% |
| ColBERT | +30ms | +10-15% |
| LLM reranking | +2000ms | +20-30% |

### Decision: BGE-reranker-base (CrossEncoder)
```python
class BGEReranker:
    def rerank(self, query: str, chunks: list, top_k: int = 5):
        # Score each query-document pair
        pairs = [[query, chunk.text] for chunk in chunks]
        scores = self.model.predict(pairs)
        
        # Sort by score, return top_k
        ranked = sorted(zip(chunks, scores), key=lambda x: -x[1])
        return ranked[:top_k]
```

### Why BGE-reranker
- Same model family as embeddings (consistent behavior)
- Best speed/quality tradeoff
- Small model (400MB) fits in GPU memory alongside embeddings

### Bi-encoder vs Cross-encoder

**Bi-encoder** (for retrieval):
```python
# Encode separately, compare with cosine
q_vec = encode(query)        # Once
d_vec = encode(document)     # Once per doc
score = cosine(q_vec, d_vec) # Fast comparison
```

**Cross-encoder** (for reranking):
```python
# Encode together, get relevance score
score = cross_encoder.predict([query, document])  # Slow but accurate
```

### Interview Answer
> "I added a reranking stage using BGE-reranker-base, a cross-encoder model. We retrieve top-20 candidates with fast bi-encoder search, then rerank with the cross-encoder which scores query-document pairs together - more accurate because of cross-attention between query and document tokens. This adds ~50ms latency but significantly improves relevance. I chose BGE-reranker because it's in the same model family as our embeddings."

---

## Decision 9: HyDE Query Expansion

### Problem
User queries are short. Document chunks are long. Embedding space mismatch.
```
Query: "What is attention?" (3 words)

Chunk: "Self-attention allows tokens to attend to each other 
        regardless of position in the sequence. The mechanism 
        computes query, key, value vectors..." (50+ words)

These embed very differently!
```

### Solution: Hypothetical Document Embeddings (HyDE)
```python
def hyde_expand(query: str) -> str:
    prompt = f"""
    Question: {query}
    
    Write a detailed 2-3 paragraph answer to this question 
    as if it appeared in a lecture transcript.
    """
    hypothetical = ollama.generate(prompt)
    return hypothetical  # Embed THIS, not the original query
```

### Why It Works
The hypothetical answer is much closer to actual document chunks in embedding space. It's like searching with a document instead of a query.

### Benchmark Results
- **+113% NLI score** (answer faithfulness)
- +2.6% MRR improvement

### Tradeoff
Adds ~1.5 seconds latency (LLM generation). Made it **optional** for quality-critical queries.

### Interview Answer
> "HyDE, or Hypothetical Document Embeddings, addresses the query-document length mismatch. A 3-word query embeds very differently than a 200-word chunk. With HyDE, I generate a hypothetical answer first, then embed that instead. The hypothesis is closer to actual documents in embedding space. It improved our NLI faithfulness score by 113% but adds 1.5 seconds latency, so I made it optional and off by default."

---

## Decision 10: Answer Generation

### Options

| Model | Quality | Speed | Cost | Privacy |
|-------|---------|-------|------|---------|
| GPT-4 | Excellent | Medium | $$$$ | Data leaves |
| Claude | Excellent | Medium | $$$ | Data leaves |
| Llama 3.2 (local) | Very Good | Fast | Free | **Private** |
| Mixtral (local) | Good | Medium | Free | **Private** |

### Decision: Ollama with Llama 3.2

### Why
- **Runs locally**: Student data never leaves university servers
- **Fast enough**: ~2-3 seconds for answer generation
- **Good quality**: Instruction following is solid
- **Free**: No API costs for university budget

### Implementation
```python
def generate_answer(query: str, context: str) -> str:
    prompt = f"""
    Based on the following lecture transcript excerpts, answer the question.
    Cite specific parts of the transcript in your answer.
    If the speaker is identified, reference them (e.g., "As Professor X mentioned...").
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:
    """
    return ollama.generate("llama3.2:latest", prompt)
```

### Interview Answer
> "I use Ollama with Llama 3.2 for answer generation. The key requirement was data privacy - student queries and lecture content can't leave university servers. Llama 3.2 runs locally with acceptable quality. The prompt includes retrieved context with speaker attribution so answers can cite 'As Professor X mentioned...' The tradeoff versus GPT-4 is maybe 10% quality reduction, but zero cost and complete privacy."

---

## Decision 11: Multi-Tenant Architecture

### Problem
Multiple courses, departments, universities - need isolation.

### Solution: Collection-Based Isolation
```python
# Each tenant gets own Qdrant collection
rag.ingest('cs229_lecture1.wav', collection_name='stanford_cs229')
rag.ingest('cs50_lecture1.wav', collection_name='harvard_cs50')

# Queries are isolated - only searches that collection
result = rag.query('What is recursion?', collection_name='harvard_cs50')
```

### Why Collections Over Namespaces
- **Complete isolation**: Security requirement for universities
- **Clean deletion**: Drop entire tenant without affecting others
- **Per-tenant config**: Different settings per collection
- **Qdrant optimizes**: Separate indexes per collection

### Interview Answer
> "I implemented multi-tenancy using Qdrant collections. Each organization gets a separate collection, providing complete data isolation - critical for universities handling student data across departments. This also enables per-tenant configuration and clean deletion. The alternative was namespace filtering within one collection, but that risks data leakage and complicates FERPA compliance."

---

## Decision 12: Production Infrastructure

### Job Queue: Redis + RQ
```python
@dataclass
class IngestJob:
    audio_path: str
    collection_name: str
    tenant_id: str
    priority: Priority  # LOW, NORMAL, HIGH
    
@dataclass
class JobCheckpoint:
    stage: JobStage     # ASR, DIARIZATION, ALIGNMENT, CHUNKING, EMBEDDING
    data: dict          # Intermediate results
```

### Why Checkpoints

Audio ingestion takes 2-10 minutes. If it fails at step 4, restart from checkpoint, not from scratch:
```python
# Save checkpoint after each stage
checkpoint = JobCheckpoint(
    stage=JobStage.DIARIZATION_COMPLETE,
    data={'transcript': transcript, 'speaker_turns': turns}
)
redis.set(f'checkpoint:{job_id}', checkpoint.to_json())

# On failure, resume from checkpoint
checkpoint = redis.get(f'checkpoint:{job_id}')
if checkpoint.stage == JobStage.DIARIZATION_COMPLETE:
    resume_from_alignment(checkpoint.data)  # Skip ASR and diarization
```

### API: FastAPI with Rate Limiting
```python
@router.post("/query")
async def search_audio(
    request: QueryRequest,
    api_key: str = Depends(get_api_key),          # Auth
    _rate_limit: None = Depends(rate_limit_dependency("query")),  # Rate limit
) -> QueryResponse:
    # Different rate limits per tier (free, basic, premium)
```

### Interview Answer
> "For production, I built a Redis job queue with checkpoint-based recovery. Audio ingestion takes 2-10 minutes, so it must be async. If a job fails at diarization, it resumes from the ASR checkpoint - saves 5+ minutes of GPU time. The FastAPI layer handles auth via API keys with different rate limits per tenant tier, implemented using Redis sliding window counters."

---

# Part 2: Interview Q&A

## System Design Questions

### Q: "Design a system to search through 10,000 hours of lecture recordings."

**Your Answer:**

"I've actually built this system. Here's the architecture:

**Ingestion Pipeline (offline, async):**
1. Audio file → Faster-Whisper ASR → word-level transcript with timestamps
2. Same audio → NeMo diarization → speaker segments
3. Custom alignment → map words to speakers
4. Speaker-turn chunking → 256 token chunks with overlap
5. Contextual processing → LLM adds context per chunk
6. BGE-M3 embedding → dense + sparse vectors
7. Qdrant storage → hybrid index

**Query Pipeline (online, sync):**
1. User query → optional HyDE expansion
2. BGE-M3 embedding → dense + sparse vectors
3. Qdrant hybrid search → RRF fusion of dense + BM25
4. BGE reranker → re-score top-20 to top-5
5. Ollama generation → synthesize answer with citations

**Scale:**
- Single GPU: 7 queries/sec, 141ms latency
- Horizontal: Add GPU workers behind load balancer
- Qdrant: Automatic sharding for large collections

**Multi-tenant:**
- Collection per organization
- Rate limiting per API key
- Redis queue for async ingestion"

---

### Q: "How would you improve retrieval quality in a RAG system?"

**Your Answer:**

"I implemented four techniques, each with measured improvements:

1. **Contextual Retrieval** (+47% precision)
   - Prepend LLM-generated context to each chunk during ingestion
   - Context explains what the chunk is about
   - Embeddings capture document-level meaning, not just isolated text

2. **Hybrid Search** (same quality, -31% latency)
   - Combine dense semantic vectors with sparse BM25
   - Dense catches meaning ('ML' matches 'machine learning')
   - Sparse catches exact keywords ('RLHF' acronym)
   - RRF fusion combines results by rank

3. **Reranking** (+15-25% relevance)
   - Cross-encoder rescores top-20 candidates
   - More accurate than bi-encoder but slower
   - Only used on small candidate set

4. **HyDE** (+113% NLI score)
   - Generate hypothetical answer, embed that instead of query
   - Bridges query-document embedding space gap
   - Optional due to latency cost

I benchmarked each on a custom evaluation dataset and kept improvements that showed statistically significant gains."

---

### Q: "How do you handle failures in a long-running ML pipeline?"

**Your Answer:**

"I implemented checkpoint-based recovery. The ingestion pipeline has 6 stages:
```
ASR → Diarization → Alignment → Chunking → Context → Embedding
```

After each stage, I persist intermediate results to Redis:
```python
checkpoint = JobCheckpoint(
    stage=JobStage.DIARIZATION_COMPLETE,
    data={'transcript': transcript, 'speaker_turns': turns}
)
redis.set(f'checkpoint:{job_id}', checkpoint.to_json())
```

If the job fails at stage 4, the worker picks up from the last checkpoint:
```python
checkpoint = redis.get(f'checkpoint:{job_id}')
if checkpoint.stage == JobStage.DIARIZATION_COMPLETE:
    # Skip ASR and diarization, resume from alignment
    resume_from_alignment(checkpoint.data)
```

This saves 5+ minutes of GPU time on failures. I also have:
- Dead letter queues for jobs that fail repeatedly
- Alerting on failure rate spikes
- Automatic retry with exponential backoff"

---

### Q: "How do you ensure thread safety with CUDA models?"

**Your Answer:**

"CUDA models aren't thread-safe - you can't load the same model in multiple threads without memory conflicts. I use a singleton pattern with mutex locking:
```python
# Global singleton
_pipeline: AudioRAG | None = None
_lock = threading.Lock()

def get_pipeline() -> AudioRAG:
    global _pipeline
    if _pipeline is None:
        _pipeline = AudioRAG(config)  # Load models once
    return _pipeline

def query(text: str) -> Result:
    with _lock:  # Serialize access to GPU
        return get_pipeline().query(text)
```

For higher throughput, I'd use:
- Multiple GPU workers (processes, not threads)
- Each process has its own model instance
- Load balancer distributes requests
- In production: Ray Serve or Triton Inference Server for proper model serving

The current single-GPU setup handles 7 qps with 141ms latency, which is sufficient for our university use case."

---

## ML/NLP Questions

### Q: "Explain how your embedding model works."

**Your Answer:**

"I use BGE-M3, which produces both dense and sparse vectors from the same model:

**Dense vectors (1024 dimensions):**
- Standard transformer encoding
- The [CLS] token embedding captures semantic meaning
- Good for similarity search - 'machine learning' close to 'ML'

**Sparse vectors (vocabulary-sized):**
- Term weights learned during training
- Like learned BM25 - each dimension is a vocabulary token
- Most are zero, only tokens in text have non-zero weights
```python
result = model.encode(text, return_sparse=True)

# Dense: semantic representation
result.dense = [0.12, -0.34, ...]  # 1024 floats

# Sparse: keyword weights  
result.sparse = {2481: 0.85, 9923: 0.42, ...}  # token_id: weight
```

I store both in Qdrant and search both indexes in parallel, combining results with Reciprocal Rank Fusion. This gives us semantic understanding AND exact keyword matching."

---

### Q: "What's the difference between bi-encoder and cross-encoder?"

**Your Answer:**

"They're both transformer-based but work differently:

**Bi-encoder** (used for retrieval):
```python
# Encode query and document SEPARATELY
q_vec = encode(query)         # Compute once
d_vec = encode(document)      # Compute once per doc
score = cosine(q_vec, d_vec)  # Fast dot product
```
- **Pros:** Fast - encode documents offline, only encode query at runtime
- **Cons:** No cross-attention between query and document
- **Use:** Initial retrieval of top-N candidates

**Cross-encoder** (used for reranking):
```python
# Encode query and document TOGETHER
score = cross_encoder.predict([query, document])
```
- **Pros:** More accurate - full attention between query and document tokens
- **Cons:** Slow - must run for each query-document pair
- **Use:** Rerank small candidate set (top-20 → top-5)

In my pipeline:
1. Bi-encoder (BGE-M3) retrieves top-20 candidates fast
2. Cross-encoder (BGE-reranker) reranks to top-5 accurately

Best of both worlds: speed from bi-encoder, accuracy from cross-encoder."

---

### Q: "How do you evaluate RAG system quality?"

**Your Answer:**

"I built a comprehensive evaluation framework with multiple metrics:

**Retrieval Metrics:**
| Metric | What It Measures |
|--------|------------------|
| Precision@K | Fraction of retrieved docs that are relevant |
| Recall@K | Fraction of relevant docs that were retrieved |
| MRR | How high is the first relevant result? |
| NDCG | Graded relevance with position discount |
| Hit Rate | Did we find at least one relevant doc? |

**Generation Metrics:**
| Metric | What It Measures |
|--------|------------------|
| Answer Similarity | Semantic similarity to ground truth |
| NLI Score | Does answer logically follow from context? (faithfulness) |
| BLEU | Lexical overlap with ground truth |

**Evaluation Dataset:**
```python
EvalSample(
    question='What is gradient descent?',
    ground_truth='An iterative optimization algorithm...',
    ground_truth_contexts=['gradient', 'optimization', 'learning rate']
)
```

I run this after every major change to catch regressions. For example, contextual retrieval improved MRR from 0.650 to 0.875 - that's how I validated it was worth the extra ingestion cost."

---

## Behavioral Questions

### Q: "Tell me about a technical decision where you had to make tradeoffs."

**Your Answer:**

"Contextual Retrieval is the best example. It improves precision by 47%, but has real costs:

**Costs:**
- Ingestion time: 10x slower (LLM call per chunk)
- Compute: Need Ollama running during ingestion
- Storage: Slightly larger chunks with context prepended
- Requires re-indexing: Can't retroactively add context

**Benefits:**
- 47% better precision
- 35% better MRR
- Better answer quality downstream

**My solution:** Make it optional.
```python
# Fast ingestion for testing
rag.ingest('lecture.wav')

# Quality ingestion for production
rag.ingest('lecture.wav', enable_contextual=True)
```

The tradeoff was worth it because:
1. Ingestion is one-time, queries happen constantly
2. Paying 10x at ingestion for 47% better queries is good ROI
3. Users can choose based on their needs"

---

### Q: "How did you decide what to build vs. use existing tools?"

**Your Answer:**

"My rule: Use existing tools for commoditized problems, build custom for differentiation.

**Used Existing:**
| Tool | Why |
|------|-----|
| Whisper | State-of-the-art ASR, no value in rebuilding |
| Qdrant | Production-grade vector DB, handles scaling |
| Redis | Reliable queue, battle-tested |
| Ollama | Simple local LLM inference |
| FastAPI | Async, type-safe, good ecosystem |

**Built Custom:**
| Component | Why |
|-----------|-----|
| Word-to-speaker alignment | No library does this well |
| Speaker-turn chunking | Domain-specific requirement |
| Contextual retrieval pipeline | Novel technique, core differentiator |
| Evaluation framework | Specific to our metrics |
| Hybrid search orchestration | Combines multiple pieces our way |

The result: I shipped faster by leveraging existing infrastructure, but the core RAG improvements are custom and defensible."

---

### Q: "What would you do differently if starting over?"

**Your Answer:**

"Three things:

1. **Start with evaluation earlier.** I built the evaluation framework in Phase 3b. Should have been Phase 1. Without metrics, I was guessing about improvements.

2. **Streaming ASR from the start.** Currently batch-only. Adding real-time is significant refactoring. Would have architected for it upfront.

3. **Better chunking experimentation.** I settled on speaker-turn chunking quickly. Could have tried semantic chunking, sliding window, or learned chunking boundaries. The 256-token size was empirical but not rigorously optimized.

That said, the iterative approach worked. Ship something, measure, improve. The evaluation framework let me quantify every change after I added it."

---

# Part 3: Real-Time ASR Integration

## Current State: Batch Processing Only

The current pipeline processes complete audio files:
```
Audio File → Whisper → Complete Transcript
```

Latency: ~10-30 seconds per minute of audio (depends on GPU).

---

## Option 1: Chunked Streaming (Simplest)

Process audio in chunks as it arrives:
```python
from faster_whisper import WhisperModel
import numpy as np
import queue
import sounddevice as sd

class RealtimeASR:
    def __init__(self, chunk_seconds: float = 5.0):
        self.model = WhisperModel("large-v3", device="cuda")
        self.audio_queue = queue.Queue()
        self.buffer = np.array([], dtype=np.float32)
        self.sample_rate = 16000
        self.chunk_seconds = chunk_seconds
        
    def audio_callback(self, indata, frames, time, status):
        """Called by sounddevice for each audio chunk."""
        self.audio_queue.put(indata.copy())
    
    def process_stream(self):
        """Process audio buffer when it reaches chunk_seconds."""
        while True:
            # Accumulate audio
            chunk = self.audio_queue.get()
            self.buffer = np.append(self.buffer, chunk.flatten())
            
            # Process when buffer is long enough
            if len(self.buffer) >= self.sample_rate * self.chunk_seconds:
                segments, _ = self.model.transcribe(
                    self.buffer,
                    language="en",
                    vad_filter=True,
                )
                
                for segment in segments:
                    yield {
                        "text": segment.text,
                        "start": segment.start,
                        "end": segment.end,
                    }
                
                # Keep last 1 second for context overlap
                self.buffer = self.buffer[-self.sample_rate:]
    
    def start_microphone(self):
        """Start streaming from microphone."""
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.audio_callback,
        ):
            for result in self.process_stream():
                print(f"[{result['start']:.1f}s] {result['text']}")
```

**Latency:** 5-7 seconds (chunk duration + processing)
**Quality:** Same as batch (Whisper large-v3)
**Complexity:** Low

---

## Option 2: WebSocket API for Browser
```python
from fastapi import FastAPI, WebSocket
from faster_whisper import WhisperModel
import numpy as np

app = FastAPI()
model = WhisperModel("large-v3", device="cuda")

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    
    buffer = np.array([], dtype=np.float32)
    
    while True:
        # Receive audio chunk from browser
        data = await websocket.receive_bytes()
        chunk = np.frombuffer(data, dtype=np.float32)
        buffer = np.append(buffer, chunk)
        
        # Process every 5 seconds of audio
        if len(buffer) >= 16000 * 5:
            segments, _ = model.transcribe(buffer)
            
            for segment in segments:
                await websocket.send_json({
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end,
                })
            
            # Keep 1s overlap
            buffer = buffer[-16000:]
```

**Browser JavaScript:**
```javascript
// Get microphone access
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
const ws = new WebSocket('ws://localhost:8000/ws/transcribe');

// Process audio
const audioContext = new AudioContext({ sampleRate: 16000 });
const source = audioContext.createMediaStreamSource(stream);
const processor = audioContext.createScriptProcessor(4096, 1, 1);

processor.onaudioprocess = (e) => {
    const audioData = e.inputBuffer.getChannelData(0);
    ws.send(new Float32Array(audioData).buffer);
};

source.connect(processor);
processor.connect(audioContext.destination);

// Receive transcripts
ws.onmessage = (event) => {
    const result = JSON.parse(event.data);
    document.getElementById('transcript').innerText += result.text + ' ';
};
```

---

## Option 3: Whisper.cpp (Lower Latency)

C++ implementation, faster than Python:
```bash
# Install
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
make stream

# Download model
./models/download-ggml-model.sh large-v3

# Run real-time from microphone
./stream -m models/ggml-large-v3.bin \
    --step 5000 \      # Process every 5000ms
    --length 5000 \    # Context length 5000ms
    --threads 4
```

**Latency:** 2-3 seconds
**Quality:** Slightly lower than Python Whisper
**Complexity:** Medium (C++ integration)

---

## Option 4: Cloud APIs (Lowest Latency)

For true real-time (<500ms), use specialized streaming services:

### Deepgram
```python
from deepgram import Deepgram
import asyncio

DEEPGRAM_API_KEY = "your-key"

async def transcribe_stream(audio_stream):
    dg = Deepgram(DEEPGRAM_API_KEY)
    
    socket = await dg.transcription.live({
        "model": "nova-2",
        "language": "en",
        "smart_format": True,
        "diarize": True,  # Speaker labels!
    })
    
    socket.on("transcript", lambda result: 
        print(result["channel"]["alternatives"][0]["transcript"])
    )
    
    async for chunk in audio_stream:
        socket.send(chunk)
```

**Latency:** ~300ms
**Quality:** Very good (Nova-2)
**Cost:** $0.0043/minute
**Diarization:** Yes, built-in

### AssemblyAI
```python
import assemblyai as aai

aai.settings.api_key = "your-key"

def on_data(transcript: aai.RealtimeTranscript):
    if transcript.text:
        print(transcript.text)

transcriber = aai.RealtimeTranscriber(
    sample_rate=16000,
    on_data=on_data,
)

transcriber.connect()
# Stream audio...
```

**Latency:** ~500ms
**Quality:** Very good
**Cost:** $0.006/minute

---

## Recommended: Hybrid Approach
```python
# audio_rag/asr/streaming.py

from enum import Enum
from typing import AsyncIterator, Callable
import numpy as np

class StreamingBackend(Enum):
    WHISPER_CHUNKED = "whisper_chunked"  # Self-hosted, 5-7s latency
    DEEPGRAM = "deepgram"                 # Cloud, 300ms latency

class StreamingASR:
    """Unified streaming ASR interface."""
    
    def __init__(
        self, 
        backend: StreamingBackend = StreamingBackend.WHISPER_CHUNKED,
        chunk_seconds: float = 5.0,
    ):
        self.backend = backend
        self.chunk_seconds = chunk_seconds
        
        if backend == StreamingBackend.WHISPER_CHUNKED:
            from faster_whisper import WhisperModel
            self.model = WhisperModel("large-v3", device="cuda")
        elif backend == StreamingBackend.DEEPGRAM:
            from deepgram import Deepgram
            self.client = Deepgram(os.environ["DEEPGRAM_API_KEY"])
    
    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        on_transcript: Callable[[dict], None],
    ):
        """Process streaming audio, emit transcripts."""
        
        if self.backend == StreamingBackend.WHISPER_CHUNKED:
            await self._whisper_stream(audio_stream, on_transcript)
        elif self.backend == StreamingBackend.DEEPGRAM:
            await self._deepgram_stream(audio_stream, on_transcript)
    
    async def _whisper_stream(self, audio_stream, on_transcript):
        buffer = np.array([], dtype=np.float32)
        
        async for chunk in audio_stream:
            audio = np.frombuffer(chunk, dtype=np.float32)
            buffer = np.append(buffer, audio)
            
            if len(buffer) >= 16000 * self.chunk_seconds:
                segments, _ = self.model.transcribe(buffer)
                
                for seg in segments:
                    on_transcript({
                        "text": seg.text,
                        "start": seg.start,
                        "end": seg.end,
                        "words": [{"word": w.word, "start": w.start, "end": w.end} 
                                  for w in (seg.words or [])],
                    })
                
                buffer = buffer[-16000:]  # Keep 1s overlap
    
    async def _deepgram_stream(self, audio_stream, on_transcript):
        socket = await self.client.transcription.live({
            "model": "nova-2",
            "language": "en",
            "smart_format": True,
        })
        
        socket.on("transcript", lambda r: on_transcript({
            "text": r["channel"]["alternatives"][0]["transcript"],
            "start": r.get("start", 0),
            "end": r.get("end", 0),
        }))
        
        async for chunk in audio_stream:
            socket.send(chunk)
```

---

## Integration with RAG Pipeline
```python
class AudioRAG:
    async def ingest_realtime(
        self,
        audio_stream: AsyncIterator[bytes],
        collection_name: str,
        streaming_backend: StreamingBackend = StreamingBackend.WHISPER_CHUNKED,
    ):
        """Stream audio directly into RAG pipeline."""
        
        streamer = StreamingASR(backend=streaming_backend)
        segments = []
        
        def on_transcript(result):
            segments.append(TranscriptSegment(
                text=result["text"],
                start=result["start"],
                end=result["end"],
                speaker=None,  # No diarization in real-time (yet)
            ))
            
            # Batch process every 10 segments
            if len(segments) >= 10:
                self._process_batch(segments, collection_name)
                segments.clear()
        
        await streamer.transcribe_stream(audio_stream, on_transcript)
        
        # Process remaining
        if segments:
            self._process_batch(segments, collection_name)
    
    def _process_batch(self, segments, collection_name):
        """Chunk, embed, store a batch of segments."""
        chunks = self.chunker.chunk(segments)
        embeddings = self.embedder.embed([c.text for c in chunks])
        self.retriever.add(chunks, embeddings, collection_name)
```

---

## Summary: Real-Time Options

| Approach | Latency | Quality | Cost | Diarization |
|----------|---------|---------|------|-------------|
| Whisper chunked | 5-7s | Best | Free | No* |
| Whisper.cpp | 2-3s | Good | Free | No |
| Deepgram | 300ms | Very Good | $0.004/min | Yes |
| AssemblyAI | 500ms | Very Good | $0.006/min | Yes |

*Real-time diarization is hard. Deepgram does it, Whisper doesn't.

**Recommendation:**
1. Start with Whisper chunked (free, good quality)
2. Add Deepgram as optional backend for live events needing low latency
3. Post-process recordings with full pipeline (NeMo diarization)

---

# Part 4: Quick Reference Cheatsheet

## Architecture at a Glance
```
INGESTION:
Audio → Whisper → NeMo → Align → Chunk → Context → BGE-M3 → Qdrant
         ASR      Diar   Words   256tok   LLM      Dense+   Hybrid
                         to Spk          Context   Sparse   Index

QUERY:
Query → (HyDE) → BGE-M3 → Hybrid → Rerank → Ollama → Answer
         Expand   Embed    RRF     Top-5    Generate
```

## Key Numbers

| Metric | Value |
|--------|-------|
| Query latency (warm) | 141ms |
| Throughput | 7.1 qps |
| Contextual improvement | +47% precision |
| Hybrid speedup | -31% latency |
| HyDE improvement | +113% NLI |

## Config Options
```python
# Best quality (slower ingestion)
rag.ingest('audio.wav', enable_contextual=True)

# Best speed
rag.query('question', search_type='hybrid', enable_hyde=False)

# Best answer quality
rag.query('question', search_type='hybrid', enable_hyde=True, generate_answer=True)
```

## Commands
```bash
# Start services
docker run -d -p 6333:6333 qdrant/qdrant
docker run -d -p 6379:6379 redis
ollama pull llama3.2:latest

# Ingest
uv run python -m audio_rag.cli ingest lecture.mp3 --collection cs229

# Query
uv run python -m audio_rag.cli query "What is ML?" --collection cs229

# API
uvicorn audio_rag.api:create_app --factory --port 8000
```

## Interview One-Liners

| Topic | One-Liner |
|-------|-----------|
| **Contextual Retrieval** | "LLM adds context to chunks during ingestion → +47% precision" |
| **Hybrid Search** | "Dense + BM25 with RRF fusion → same quality, 31% faster" |
| **HyDE** | "Generate hypothetical answer, embed that → bridges query-doc gap" |
| **Reranking** | "Cross-encoder rescores top-20 → more accurate than bi-encoder" |
| **Multi-tenant** | "Qdrant collections → complete data isolation per org" |
| **Checkpoints** | "Redis persists after each stage → resume on failure" |

---

## Final Words

This project demonstrates:
- **Research implementation** (Contextual Retrieval, HyDE, RRF)
- **Production engineering** (queues, checkpoints, multi-tenant)
- **Quantified improvements** (47% precision, 31% latency)
- **End-to-end system** (10+ components working together)

You can talk about this for 30+ minutes in any technical interview. The key is connecting decisions to measurable outcomes.

Good luck! 🚀
