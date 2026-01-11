#!/bin/bash
# Quick start script for demo recording

echo "🚀 Starting Audio RAG Demo Environment..."

# Check Docker
if ! docker ps > /dev/null 2>&1; then
    echo "❌ Docker not running. Start Docker first."
    exit 1
fi

# Start Qdrant
echo "📦 Starting Qdrant..."
docker start qdrant 2>/dev/null || docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
sleep 2

# Start Redis
echo "📦 Starting Redis..."
docker start redis 2>/dev/null || docker run -d -p 6379:6379 --name redis redis
sleep 2

# Check Ollama
echo "🤖 Checking Ollama..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Ollama not running. Start it with: ollama serve"
fi

# Start API
echo "🌐 Starting API server..."
cd ~/projects/audio-rag
pkill -f "uvicorn audio_rag" 2>/dev/null
uv run uvicorn audio_rag.api:create_app --factory --host 0.0.0.0 --port 8000 &
sleep 5

# Start Frontend
echo "🎨 Starting Frontend..."
cd ~/projects/audio-rag/frontend
pkill -f "vite" 2>/dev/null
npm run dev -- --host &
sleep 3

echo ""
echo "✅ Demo environment ready!"
echo ""
echo "   Frontend: http://localhost:3000"
echo "   API:      http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop recording, then run: ./scripts/demo_stop.sh"
