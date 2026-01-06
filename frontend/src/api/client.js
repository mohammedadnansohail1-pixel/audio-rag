import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const API_KEY = import.meta.env.VITE_API_KEY || 'dev-key-12345';

const client = axios.create({
  baseURL: API_BASE,
  headers: {
    'X-API-Key': API_KEY,
  },
});

// Query audio content
export async function searchAudio(query, options = {}) {
  const response = await client.post('/api/v1/query', {
    query,
    collection_name: options.collection || 'audio_rag_contextual',
    top_k: options.topK || 5,
    search_type: options.searchType || 'hybrid',
    enable_reranking: options.reranking ?? true,
    generate_answer: options.generateAnswer ?? true,
  });
  return response.data;
}

// Get collection info
export async function getCollectionCount(collection) {
  const response = await client.get(`/api/v1/collections/${collection}/count`);
  return response.data;
}

// List collections (via Qdrant directly for now)
export async function listCollections() {
  try {
    const response = await axios.get('http://localhost:6333/collections');
    return response.data.result.collections.map(c => c.name);
  } catch {
    return ['audio_rag_contextual', 'audio_rag_hybrid'];
  }
}

// Ingest audio (multipart form)
export async function ingestAudio(file, collection, options = {}) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('collection_name', collection);
  if (options.priority) formData.append('priority', options.priority);
  
  const response = await client.post('/api/v1/ingest', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
}

// Get job status
export async function getJobStatus(jobId) {
  const response = await client.get(`/api/v1/jobs/${jobId}`);
  return response.data;
}

// Streaming WebSocket URL
export function getStreamingUrl(options = {}) {
  const params = new URLSearchParams();
  if (options.language) params.set('language', options.language);
  if (options.chunkDuration) params.set('chunk_duration', options.chunkDuration);
  
  const wsBase = API_BASE.replace('http', 'ws');
  return `${wsBase}/api/v1/ws/transcribe?${params}`;
}

export default client;
