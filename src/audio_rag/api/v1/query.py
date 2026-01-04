"""Query/search endpoint."""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from audio_rag.api.deps import (
    Context,
    Queue,
    get_request_context,
    get_queue,
    rate_limit_dependency,
)
from audio_rag.api.schemas import QueryRequest, QueryResponse, SearchResult

logger = logging.getLogger(__name__)

router = APIRouter()


def get_qdrant_client(request: Request) -> QdrantClient:
    """Get Qdrant client from app state."""
    client = getattr(request.app.state, "qdrant_client", None)
    if not client:
        from audio_rag.api.config import DEFAULT_API_CONFIG
        client = QdrantClient(
            host=DEFAULT_API_CONFIG.qdrant_host,
            port=DEFAULT_API_CONFIG.qdrant_port,
        )
        request.app.state.qdrant_client = client
    return client


async def embed_query(query: str, request: Request) -> list[float]:
    """Get embedding for query text."""
    embedder = getattr(request.app.state, "embedder", None)
    
    if embedder is None:
        try:
            from audio_rag.embeddings import EmbeddingModel
            from audio_rag.config import EmbeddingConfig
            
            config = EmbeddingConfig()
            embedder = EmbeddingModel(config)
            request.app.state.embedder = embedder
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"error": "Embedding model unavailable", "code": "SERVICE_UNAVAILABLE"},
            )
    
    try:
        embeddings = embedder.embed([query])
        return embeddings[0].tolist()
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to embed query", "code": "EMBEDDING_ERROR"},
        )


def build_qdrant_filter(filters: dict[str, Any] | None) -> dict | None:
    """Build Qdrant filter from user-provided filters."""
    if not filters:
        return None
    
    conditions = []
    for key, value in filters.items():
        if isinstance(value, list):
            conditions.append({"key": f"metadata.{key}", "match": {"any": value}})
        else:
            conditions.append({"key": f"metadata.{key}", "match": {"value": value}})
    
    return {"must": conditions} if conditions else None


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Search audio content",
    dependencies=[Depends(rate_limit_dependency("query"))],
)
async def query_audio(
    request: Request,
    body: QueryRequest,
    ctx: Context,
):
    """Search transcribed audio content using semantic similarity."""
    start_time = time.perf_counter()
    
    collection = body.collection or ctx.tenant_id
    qdrant = get_qdrant_client(request)
    
    # Check collection exists
    try:
        collections = qdrant.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if collection not in collection_names:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": f"Collection not found: {collection}", "code": "COLLECTION_NOT_FOUND"},
            )
    except UnexpectedResponse as e:
        logger.error(f"Qdrant error checking collections: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "Search service unavailable", "code": "SERVICE_UNAVAILABLE"},
        )
    
    query_vector = await embed_query(body.query, request)
    qdrant_filter = build_qdrant_filter(body.filters)
    
    try:
        search_results = qdrant.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=body.limit + body.offset,
            score_threshold=body.score_threshold,
            query_filter=qdrant_filter,
            with_payload=True,
        )
    except UnexpectedResponse as e:
        logger.error(f"Qdrant search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "Search failed", "code": "SEARCH_ERROR"},
        )
    
    search_results = search_results[body.offset:]
    
    results = []
    for hit in search_results:
        payload = hit.payload or {}
        
        result = SearchResult(
            chunk_id=str(hit.id),
            text=payload.get("text", ""),
            score=hit.score,
            start_time=payload.get("start_time"),
            end_time=payload.get("end_time"),
            speaker=payload.get("speaker"),
            metadata=payload.get("metadata", {}),
        )
        
        if body.include_context:
            result.context_before = payload.get("context_before")
            result.context_after = payload.get("context_after")
        
        results.append(result)
    
    query_time = (time.perf_counter() - start_time) * 1000
    
    logger.info(f"Query completed: collection={collection}, results={len(results)}, time={query_time:.1f}ms")
    
    return QueryResponse(
        results=results,
        total=len(search_results),
        query=body.query,
        collection=collection,
        query_time_ms=round(query_time, 2),
    )
