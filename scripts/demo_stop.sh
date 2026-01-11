#!/bin/bash
# Stop demo environment

echo "🛑 Stopping Audio RAG Demo Environment..."

pkill -f "uvicorn audio_rag" 2>/dev/null
pkill -f "vite" 2>/dev/null

echo "✅ Servers stopped. Docker containers still running (qdrant, redis)."
echo "   To stop Docker: docker stop qdrant redis"
