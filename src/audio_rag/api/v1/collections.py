"""Collections management endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from audio_rag.utils import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["collections"])


class CollectionInfo(BaseModel):
    name: str
    count: int


@router.get("/collections")
async def list_collections() -> list[str]:
    """List all available collections."""
    try:
        from qdrant_client import QdrantClient
        
        client = QdrantClient(host="localhost", port=6333)
        collections = client.get_collections()
        
        return [c.name for c in collections.collections]
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        # Return defaults if Qdrant unavailable
        return ["audio_rag_contextual", "audio_rag_hybrid"]


@router.get("/collections/{collection_name}")
async def get_collection_info(collection_name: str) -> CollectionInfo:
    """Get collection info including document count."""
    try:
        from qdrant_client import QdrantClient
        
        client = QdrantClient(host="localhost", port=6333)
        info = client.get_collection(collection_name)
        
        return CollectionInfo(
            name=collection_name,
            count=info.points_count,
        )
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        raise HTTPException(status_code=404, detail=f"Collection not found: {collection_name}")
