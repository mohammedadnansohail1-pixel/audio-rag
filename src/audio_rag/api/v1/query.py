"""Query endpoint for searching audio content."""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Literal

from audio_rag.api.deps import get_api_key, rate_limit_dependency
from audio_rag.api.schemas import ErrorResponse
from audio_rag.pipeline import QueryPipeline, QueryResult
from audio_rag.config import load_config
from audio_rag.core import PipelineError, RetrievalError
from audio_rag.utils import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["query"])

_query_pipeline: QueryPipeline | None = None


def get_query_pipeline() -> QueryPipeline:
    """Get or create query pipeline singleton."""
    global _query_pipeline
    if _query_pipeline is None:
        config = load_config()
        _query_pipeline = QueryPipeline(config)
    return _query_pipeline


class QueryRequest(BaseModel):
    """Query request body."""
    query: str = Field(..., min_length=1, max_length=2000, description="Search query text")
    collection_name: str = Field(
        default="audio_rag",
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Collection to search (tenant ID)",
    )
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    filter_metadata: dict | None = Field(default=None, description="Optional metadata filter")
    
    # New advanced features
    search_type: Literal["dense", "sparse", "hybrid"] = Field(
        default="hybrid", description="Search type: dense (semantic), sparse (BM25), or hybrid (RRF fusion)"
    )
    enable_hyde: bool = Field(default=False, description="Enable HyDE query expansion")
    enable_reranking: bool = Field(default=True, description="Enable BGE reranking")
    generate_answer: bool = Field(default=False, description="Generate LLM answer from context")
    include_context: bool = Field(default=False, description="Include LLM-formatted context")


class ChunkResult(BaseModel):
    """Single search result chunk."""
    text: str
    speaker: str | None
    start: float
    end: float
    score: float
    source: str | None
    metadata: dict | None


class QueryResponse(BaseModel):
    """Query response body."""
    query: str
    collection_name: str
    results: list[ChunkResult]
    result_count: int
    context: str | None = None
    generated_answer: str | None = None
    search_type: str
    reranked: bool
    hyde_used: bool


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        429: {"model": ErrorResponse, "description": "Rate limited"},
        500: {"model": ErrorResponse, "description": "Query failed"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    summary="Search audio content",
    description="Search ingested audio content using semantic similarity with advanced RAG features.",
)
async def search_audio(
    request: QueryRequest,
    api_key: str = Depends(get_api_key),
    _rate_limit: None = Depends(rate_limit_dependency("query")),
) -> QueryResponse:
    """Search ingested audio content with advanced RAG features.

    Features:
    - Hybrid search (dense + sparse BM25)
    - HyDE query expansion
    - BGE reranking
    - LLM answer generation
    """
    try:
        pipeline = get_query_pipeline()

        result: QueryResult = pipeline.query(
            query_text=request.query,
            collection_name=request.collection_name,
            top_k=request.top_k,
            filter_metadata=request.filter_metadata,
            search_type=request.search_type,
            enable_hyde=request.enable_hyde,
            enable_reranking=request.enable_reranking,
            generate_answer=request.generate_answer,
        )

        chunks = [
            ChunkResult(
                text=r.chunk.text,
                speaker=r.chunk.speaker,
                start=r.chunk.start,
                end=r.chunk.end,
                score=r.score,
                source=r.source,
                metadata=r.chunk.metadata,
            )
            for r in result.results
        ]

        context = None
        if request.include_context and result.results:
            context = pipeline.get_context_for_llm(
                query=request.query,
                collection_name=request.collection_name,
                top_k=request.top_k,
                filter_metadata=request.filter_metadata,
            )

        return QueryResponse(
            query=result.query,
            collection_name=result.collection_name,
            results=chunks,
            result_count=len(chunks),
            context=context,
            generated_answer=result.generated_answer,
            search_type=result.search_type,
            reranked=result.reranked,
            hyde_used=result.hyde_used,
        )

    except RetrievalError as e:
        logger.error(f"Retrieval error: {e}")
        if "connect" in str(e).lower() or "connection" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Vector store unavailable: {e}",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {e}",
        )

    except PipelineError as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query pipeline error: {e}",
        )

    except Exception as e:
        logger.exception(f"Unexpected error in query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get(
    "/collections/{collection_name}/count",
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Collection not found"},
        500: {"model": ErrorResponse, "description": "Failed to get count"},
    },
    summary="Get collection document count",
)
async def get_collection_count(
    collection_name: str,
    api_key: str = Depends(get_api_key),
) -> dict:
    """Get the number of chunks in a collection."""
    try:
        pipeline = get_query_pipeline()
        if not pipeline.retriever.collection_exists(collection_name):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{collection_name}' not found",
            )
        count = pipeline.retriever.count(collection_name)
        return {"collection_name": collection_name, "count": count}
    except HTTPException:
        raise
    except RetrievalError as e:
        logger.error(f"Failed to get count: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get collection count: {e}",
        )


@router.delete(
    "/collections/{collection_name}",
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Collection not found"},
        500: {"model": ErrorResponse, "description": "Failed to delete"},
    },
    summary="Delete a collection",
)
async def delete_collection(
    collection_name: str,
    api_key: str = Depends(get_api_key),
) -> dict:
    """Delete an entire collection and all its data."""
    try:
        pipeline = get_query_pipeline()
        if not pipeline.retriever.collection_exists(collection_name):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{collection_name}' not found",
            )
        pipeline.retriever.delete_collection(collection_name)
        logger.info(f"Deleted collection: {collection_name}")
        return {"message": f"Collection '{collection_name}' deleted", "collection_name": collection_name}
    except HTTPException:
        raise
    except RetrievalError as e:
        logger.error(f"Failed to delete collection: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete collection: {e}",
        )
