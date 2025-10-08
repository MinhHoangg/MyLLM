"""
API Routes for Search functionality
"""
from fastapi import APIRouter, HTTPException
from backend.schemas import SearchRequest, SearchResponse, DocumentMetadata
from backend.app_state import state
from typing import List
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/search", tags=["search"])

@router.post("/", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search documents in the vector database.
    Returns ALL relevant documents above similarity threshold.
    """
    try:
        if not state.vector_db:
            raise HTTPException(status_code=500, detail="Vector database not initialized")
        
        logger.info(f"Searching for: {request.query}, threshold: {request.similarity_threshold}")
        
        # Perform search with unlimited results
        results = state.vector_db.search(
            query=request.query,
            similarity_threshold=request.similarity_threshold
        )
        
        # Filter by content_type if specified
        if request.content_type:
            results = [
                r for r in results
                if r.get("metadata", {}).get("content_type") == request.content_type
            ]
        
        # Convert to DocumentMetadata format
        documents = []
        for result in results:
            documents.append(DocumentMetadata(
                id=result.get("id", ""),
                content=result.get("document", ""),
                metadata=result.get("metadata", {}),
                similarity=result.get("similarity")
            ))
        
        logger.info(f"Found {len(documents)} relevant documents")
        
        return SearchResponse(
            query=request.query,
            documents=documents,
            total_results=len(documents),
            similarity_threshold=request.similarity_threshold,
            metadata={
                "content_type_filter": request.content_type,
                "search_type": "multimodal_clip"
            }
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
