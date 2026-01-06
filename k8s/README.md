# Audio RAG Kubernetes Deployment

Production-ready Kubernetes deployment using Helm.

## Prerequisites

- Kubernetes 1.25+
- Helm 3.0+
- NVIDIA GPU Operator (for GPU workers)
- Storage class with ReadWriteMany support (for shared volumes)

## Quick Start
```bash
# Add to your cluster
cd k8s/helm

# Install with default values
helm install audio-rag ./audio-rag -n audio-rag --create-namespace

# Install with custom values
helm install audio-rag ./audio-rag -n audio-rag --create-namespace \
  --set secrets.apiKey=your-secure-key \
  --set api.ingress.hosts[0].host=audio-rag.yourdomain.com
```

## Configuration

### Minimal Production Setup
```yaml
# values-prod.yaml
api:
  replicaCount: 3
  resources:
    requests:
      cpu: 1000m
      memory: 2Gi

worker:
  replicaCount: 2
  resources:
    requests:
      nvidia.com/gpu: 1

secrets:
  apiKey: "your-secure-api-key-here"

api:
  ingress:
    enabled: true
    hosts:
      - host: audio-rag.yourdomain.com
        paths:
          - path: /
            pathType: Prefix
    tls:
      - secretName: audio-rag-tls
        hosts:
          - audio-rag.yourdomain.com
```
```bash
helm install audio-rag ./audio-rag -f values-prod.yaml -n audio-rag --create-namespace
```

### University Deployment (UNT Example)
```yaml
# values-unt.yaml
api:
  replicaCount: 2
  env:
    AUDIO_RAG_ENV: production

worker:
  replicaCount: 1
  nodeSelector:
    nvidia.com/gpu: "true"

qdrant:
  persistence:
    size: 100Gi

persistence:
  uploads:
    size: 500Gi  # Lots of lecture recordings

secrets:
  apiKey: "unt-cs5500-api-key"
```

## Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                         Ingress (nginx)                         │
│                    audio-rag.yourdomain.com                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│      Frontend (2x)      │   │       API (2-10x)       │
│     React + Nginx       │   │   FastAPI + Uvicorn     │
│      Port: 80           │   │      Port: 8000         │
└─────────────────────────┘   └───────────┬─────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
        ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
        │    Redis (1x)     │ │   Qdrant (1x)     │ │   Ollama (1x)     │
        │   Job Queue       │ │  Vector Store     │ │   LLM Server      │
        │   Port: 6379      │ │   Port: 6333      │ │   Port: 11434     │
        └───────────────────┘ └───────────────────┘ └───────────────────┘
                    │
                    ▼
        ┌───────────────────┐
        │  GPU Worker (1x)  │
        │ Whisper + NeMo    │
        │ nvidia.com/gpu: 1 │
        └───────────────────┘
```

## Scaling

### Horizontal Pod Autoscaler

API pods auto-scale based on CPU:
```yaml
api:
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilization: 70
```

### Manual Worker Scaling
```bash
# Scale GPU workers
kubectl scale deployment/audio-rag-worker --replicas=3 -n audio-rag
```

## Monitoring

### Health Checks
```bash
# API health
curl http://audio-rag.local/health/live
curl http://audio-rag.local/health/ready

# Qdrant health
kubectl exec -it audio-rag-qdrant-0 -- curl localhost:6333/healthz
```

### Logs
```bash
# API logs
kubectl logs -l app.kubernetes.io/component=api -f -n audio-rag

# Worker logs
kubectl logs -l app.kubernetes.io/component=worker -f -n audio-rag
```

### Prometheus Metrics (if enabled)
```yaml
api:
  podAnnotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8000"
    prometheus.io/path: "/metrics"
```

## Persistent Volumes

| PVC | Default Size | Purpose |
|-----|--------------|---------|
| uploads | 100Gi | Audio file storage |
| models | 50Gi | ML model cache |
| redis | 8Gi | Job queue persistence |
| qdrant | 50Gi | Vector database |
| ollama | 20Gi | LLM model storage |

## Troubleshooting

### GPU Worker Not Starting
```bash
# Check GPU operator
kubectl get pods -n gpu-operator

# Check node GPU labels
kubectl get nodes -L nvidia.com/gpu

# Check worker events
kubectl describe pod -l app.kubernetes.io/component=worker -n audio-rag
```

### Qdrant Out of Memory
```yaml
qdrant:
  resources:
    limits:
      memory: 16Gi  # Increase limit
```

### API 502 Errors
```bash
# Check API pods
kubectl get pods -l app.kubernetes.io/component=api -n audio-rag

# Check ingress
kubectl describe ingress audio-rag-ingress -n audio-rag
```

## Uninstall
```bash
helm uninstall audio-rag -n audio-rag

# Remove PVCs (WARNING: deletes data)
kubectl delete pvc -l app.kubernetes.io/instance=audio-rag -n audio-rag
```
