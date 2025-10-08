"""
API Routes for Document Management
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from backend.schemas import UploadResponse, DocumentMetadata
from backend.app_state import state
from typing import List, Optional
import logging
import tempfile
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])

@router.post("/upload", response_model=UploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    use_adaptive_chunking: bool = Form(True)
):
    """
    Upload documents (text/images) to the vector database.
    Files are chunked and embedded using CLIP.
    """
    try:
        if not state.ingestion or not state.vector_db:
            raise HTTPException(status_code=500, detail="Ingestion system not initialized")
        
        # Create temp directory for uploads
        temp_dir = tempfile.mkdtemp()
        logger.info(f"📤 Processing {len(files)} uploaded files")
        
        temp_file_paths = []
        processed_files = []
        
        # Save uploaded files with original names
        for uploaded_file in files:
            temp_file_path = Path(temp_dir) / uploaded_file.filename
            with open(temp_file_path, "wb") as f:
                shutil.copyfileobj(uploaded_file.file, f)
            temp_file_paths.append(str(temp_file_path))
            processed_files.append(uploaded_file.filename)
        
        # Process files through ingestion pipeline
        chunks_processed = 0
        images_processed = 0
        results = []
        
        for file_path in temp_file_paths:
            try:
                # Ingest file (returns text_documents as List[Document], image_documents as List[Dict])
                result = state.ingestion.ingest_file(file_path=file_path)
                
                # Get documents from result
                text_docs = result.get("text_documents", [])  # List[Document]
                image_docs = result.get("image_documents", [])  # List[Dict]
                
                # Store text documents in vector DB (expects List[Document])
                if text_docs and state.vector_db:
                    logger.info(f"Adding {len(text_docs)} text documents to vector DB")
                    state.vector_db.add_text_documents(text_docs)
                
                # Store image documents in vector DB (expects List[Dict])
                if image_docs and state.vector_db:
                    logger.info(f"Adding {len(image_docs)} image documents to vector DB")
                    state.vector_db.add_image_documents(image_docs)
                
                num_chunks = result.get("num_text_chunks", 0)
                num_images = result.get("num_images", 0)
                
                chunks_processed += num_chunks
                images_processed += num_images
                
                results.append({
                    "file": Path(file_path).name,
                    "chunks": num_chunks,
                    "images": num_images,
                    "status": "success"
                })
                logger.info(f"✅ Processed {Path(file_path).name}: {num_chunks} chunks, {num_images} images stored in vector DB")
            except Exception as e:
                logger.error(f"❌ Error processing {Path(file_path).name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                results.append({"error": str(e), "file": Path(file_path).name, "status": "error"})
        
        # Cleanup temp directory
        shutil.rmtree(temp_dir)
        
        logger.info(f"🎉 Upload complete: {len(processed_files)} files, {chunks_processed} text chunks, {images_processed} images saved to vector DB")
        
        return UploadResponse(
            status="success",
            files_processed=len(processed_files),
            chunks=chunks_processed,
            images=images_processed,
            results=results
        )
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=List[DocumentMetadata])
async def list_documents(
    limit: Optional[int] = None,
    content_type: Optional[str] = None
):
    """
    List all documents in the vector database.
    Optionally filter by content_type (text/image).
    """
    try:
        if not state.vector_db:
            raise HTTPException(status_code=500, detail="Vector database not initialized")
        
        # Get all documents - returns dict with 'ids', 'documents', 'metadatas'
        db_results = state.vector_db.get_all_documents(limit=limit)
        
        ids = db_results.get('ids', [])
        documents = db_results.get('documents', [])
        metadatas = db_results.get('metadatas', [])
        
        # Convert to DocumentMetadata format
        result = []
        for i, doc_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) else {}
            doc_content = documents[i] if i < len(documents) else ""
            
            # Filter by content_type if specified
            if content_type and metadata.get("content_type") != content_type:
                continue
            
            result.append(DocumentMetadata(
                id=doc_id,
                content=doc_content,
                metadata=metadata,
                similarity=None
            ))
        
        return result
        
    except Exception as e:
        logger.error(f"List documents error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete a specific document by ID"""
    try:
        if not state.vector_db:
            raise HTTPException(status_code=500, detail="Vector database not initialized")
        
        # Delete from vector database
        state.vector_db.delete_documents([document_id])
        
        return {
            "status": "success",
            "message": f"Document {document_id} deleted",
            "deleted_id": document_id
        }
        
    except Exception as e:
        logger.error(f"Delete document error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/")
async def clear_all_documents():
    """Clear all documents from the vector database"""
    try:
        if not state.vector_db:
            raise HTTPException(status_code=500, detail="Vector database not initialized")
        
        # Get all documents
        all_docs = state.vector_db.get_all_documents()
        doc_ids = [doc.get("id") for doc in all_docs if doc.get("id")]
        
        # Delete all
        if doc_ids:
            state.vector_db.delete_documents(doc_ids)
        
        return {
            "status": "success",
            "message": f"Cleared {len(doc_ids)} documents",
            "count": len(doc_ids)
        }
        
    except Exception as e:
        logger.error(f"Clear documents error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_database_stats():
    """Get statistics about the vector database"""
    try:
        if not state.vector_db:
            raise HTTPException(status_code=500, detail="Vector database not initialized")
        
        # Get all documents - returns dict with 'ids', 'documents', 'metadatas'
        db_results = state.vector_db.get_all_documents()
        metadatas = db_results.get('metadatas', [])
        
        # Count by content type
        text_count = sum(1 for metadata in metadatas 
                        if metadata.get("content_type") == "text")
        image_count = sum(1 for metadata in metadatas 
                         if metadata.get("content_type") == "image")
        
        return {
            "total_documents": len(db_results.get('ids', [])),
            "text_documents": text_count,
            "image_documents": image_count,
            "collection_name": state.vector_db.collection_name,
            "persist_directory": str(state.vector_db.persist_directory),
            "embedding_dimension": 512  # CLIP dimension
        }
        
    except Exception as e:
        logger.error(f"Get stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
