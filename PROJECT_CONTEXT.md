# Audio RAG Project Context
**Last Updated:** 2025-01-04
**Phase:** 5 Complete (Circuit Breakers & Resilience)
**Next Phase:** 6 (LLM Integration or Production Deployment)

---

## Current Status

### Completed Phases

#### Phase 1: Queue System ✅
- Redis + RQ job queue for async audio processing
- Priority queues (high, normal, low)
- Tenant isolation (collection per course for FERPA)
- Idempotency keys to prevent duplicate jobs
- Job lifecycle: PENDING → RUNNING → COMPLETED/FAILED

**Files:**
```
src/audio_rag/queue/
├── __init__.py
├── exceptions.py      # 10+ custom exceptions
├── job.py             # IngestJob, JobResult, JobCheckpoint
├── config.py          # QueueConfig, RedisConfig, WorkerConfig
├── connection.py      # RedisConnectionManager with circuit breaker
├── validation.py      # Audio + Tenant validators
├── queue.py           # AudioRAGQueue main class
└── worker.py          # GPUWorker for ML processing
```

#### Phase 2: API Layer ✅
- FastAPI with async endpoints
- API versioning (/api/v1/)
- Rate limiting (Redis sliding window)
- API key authentication
- File upload handling

**Endpoints:**
| Method | Path | Description |
|--------|------|-------------|
| GET | /health/live | Liveness probe |
| GET | /health/ready | Readiness (checks Redis, Qdrant) |
| GET | /api/v1/ | API info |
| POST | /api/v1/ingest | Upload audio → returns job_id |
| GET | /api/v1/jobs/{id} | Job status |
| DELETE | /api/v1/jobs/{id} | Cancel job |
| POST | /api/v1/query | Search audio chunks |

**Files:**
```
src/audio_rag/api/
├── __init__.py
├── app.py             # create_app() factory
├── config.py          # APIConfig, rate limits, timeouts
├── deps.py            # Dependencies (auth, rate limit)
├── health.py          # Health check endpoints
├── middleware.py      # Logging, error handling
├── schemas.py         # Pydantic models
└── v1/
    ├── router.py
    ├── ingest.py
    ├── jobs.py
    └── query.py
```

#### Phase 3: Testing Framework ✅
- pytest + pytest-asyncio + pytest-cov
- fakeredis for Redis mocking
- 89 tests passing initially

**Test Structure:**
```
tests/
├── conftest.py
├── unit/
│   ├── queue/         # 48 tests
│   ├── api/           # 16 tests
│   └── core/          # 33 tests
└── integration/       # 18 tests
```

#### Phase 4: Docker Production Setup ✅
- Multi-stage builds (minimal images)
- Non-root user (security)
- Health checks (Kubernetes-ready)
- GPU support via NVIDIA Container Toolkit

**Files:**
```
.dockerignore
Dockerfile.api          # API server (~150MB)
Dockerfile.worker       # GPU worker (~8GB)
docker-compose.yml      # Full stack orchestration
.env.example            # Environment template
```

**Docker Commands:**
```bash
# Infrastructure only
docker compose up -d redis qdrant

# Full stack (no GPU)
docker compose up -d

# With GPU worker
docker compose --profile gpu up -d
```

#### Phase 5: Circuit Breakers & Resilience ✅
- Circuit breaker pattern for external services
- Retry with exponential backoff (tenacity)
- Fallback chains for graceful degradation
- Timeout patterns for async operations

**Resilience Patterns:**
| Pattern | Use Case | Config |
|---------|----------|--------|
| Circuit Breaker | Redis, Qdrant, HuggingFace | fail_max=5, reset=30s |
| Retry | Transient failures | 3 attempts, exponential backoff |
| Fallback | Model loading | GPU→CPU→smaller model |
| Timeout | All operations | Per-component limits |

**Files:**
```
src/audio_rag/core/resilience/
├── __init__.py
├── circuit_breaker.py  # CircuitBreaker, CircuitState
├── retry.py            # retry_with_backoff, retry_redis, retry_qdrant
├── fallback.py         # FallbackChain, has_cuda, has_gpu_memory
└── timeout.py          # async_timeout, with_timeout, TimeoutConfig
```

**Pre-configured Breakers:**
- `REDIS_BREAKER_CONFIG`: 5 failures, 30s reset
- `QDRANT_BREAKER_CONFIG`: 3 failures, 60s reset
- `HUGGINGFACE_BREAKER_CONFIG`: 3 failures, 120s reset

**Pre-built Fallback Chains:**
- `create_asr_fallback_chain()`: large-v3→medium→base→CPU
- `create_embedding_fallback_chain()`: BGE-M3 GPU→CPU

---

## Test Summary

**Total: 126 tests passing**

| Category | Count |
|----------|-------|
| Queue (job, config, validation) | 48 |
| API (schemas) | 16 |
| Resilience (circuit breaker, retry, fallback, timeout) | 33 |
| Integration (health, ingest, jobs) | 18 |
| **Total** | **126** |

---

## Architecture Overview
```
┌─────────────────────────────────────────────────────────────────┐
│                         Client                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Server (api)                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Circuit Breaker → Retry → Timeout → Rate Limit         │    │
│  └─────────────────────────────────────────────────────────┘    │
│  /api/v1/ingest  /api/v1/jobs  /api/v1/query  /health/*         │
└─────────────────────────────────────────────────────────────────┘
           │                                      │
           ▼                                      ▼
┌─────────────────────┐              ┌─────────────────────┐
│       Redis         │              │       Qdrant        │
│   - Job Queue       │              │   - Vector Store    │
│   - Rate Limits     │              │   - Collections     │
│   - Circuit State   │              │                     │
└─────────────────────┘              └─────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GPU Worker (worker)                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Fallback Chain: large-v3 → medium → base → CPU         │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ASR (Whisper) → Diarization (PyAnnote) → Embeddings (BGE-M3)   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Job Queue | Redis + RQ | Simple, Python-native, fits pilot scale |
| API Framework | FastAPI | Async, OpenAPI docs, Pydantic |
| Multi-tenancy | Collection per course | FERPA isolation |
| Auth | API keys | Simple start, OAuth later |
| Versioning | URI path (/api/v1/) | Clear, cacheable |
| Docker base | python:3.11-slim | Balance size/compatibility |
| GPU base | pytorch/pytorch:2.4.0-cuda12.1 | Pre-built PyTorch+CUDA |
| Resilience | tenacity + custom | Production-tested retry library |
| Circuit Breaker | Custom implementation | Lightweight, no external deps |

---

## Dependencies Added
```toml
# Core
redis = "^5.0"
rq = "^1.16"
fastapi = "^0.115"
uvicorn = { version = "^0.32", extras = ["standard"] }
python-multipart = "^0.0.9"
aiofiles = "^24.1"
httpx = "^0.27"

# Resilience
tenacity = "^9.0"
pybreaker = "^1.2"

# Dev
pytest = "^9.0"
pytest-asyncio = "^1.3"
pytest-cov = "^7.0"
fakeredis = "^2.25"
```

---

## Running the Project
```bash
# Start infrastructure
docker compose up -d redis qdrant

# Install dependencies
uv sync

# Run API server
uv run uvicorn audio_rag.api.app:create_app --factory --reload

# Run tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest --cov=src/audio_rag --cov-report=term-missing
```

---

## Next Steps Options

### Option A: LLM Integration (Claude API)
- Add Claude for response synthesis
- Implement RAG query pipeline
- Add streaming responses

### Option B: Production Deployment
- CI/CD with GitHub Actions
- Kubernetes manifests
- Monitoring (Prometheus + Grafana)

### Option C: Frontend
- React/Next.js UI
- Audio upload interface
- Search results display

### Option D: Worker Implementation
- Complete GPU worker integration
- Test with real audio files
- End-to-end pipeline testing

---

## Known Issues

1. **Qdrant version warning:** Client 1.16.2 vs Server 1.12.0 (minor)
2. **Deprecation warning:** HTTP_422_UNPROCESSABLE_ENTITY (cosmetic)
3. **Worker not tested:** Requires GPU + models for full testing

---

## Documentation

| File | Purpose |
|------|---------|
| `PROJECT_CONTEXT.md` | This file - project state |
| `FAILURE_AND_FUTURE_PROOFING.md` | Failure analysis, resilience patterns |
| `SCALABILITY_AND_BUSINESS.md` | Business model, pricing, scaling |

---

## Contact

**Developer:** Adnan
**Project:** Audio RAG System
**Target:** UNT Pilot (Spring 2026)
