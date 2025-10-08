"""
FastAPI Backend for Multimodal RAG Chatbot
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Optional
import uvicorn
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.chatbot import MultimodalChatbot
from app.vector_db_clip import MultimodalVectorDatabase
from app.ingestion import DocumentIngestion
from models.dnotitia_model import DNotitiaModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multimodal RAG Chatbot API",
    description="REST API for multimodal chatbot with document understanding",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import global state
from backend.app_state import state

@app.on_event("startup")
async def startup_event():
    """Initialize application components on startup"""
    logger.info("Initializing application...")
    
    try:
        # Initialize vector database
        logger.info("Loading vector database...")
        state.vector_db = MultimodalVectorDatabase(
            persist_directory="data/embeddings_db",
            collection_name="multimodal_docs",
            use_clip=True
        )
        
        # Initialize model immediately on startup
        logger.info("Loading model on startup...")
        model_size = "7.8B" if state.high_parameter else "2.4B"
        logger.info(f"Loading {model_size} model... This may take a moment.")
        state.model = DNotitiaModel(high_parameter=state.high_parameter)
        logger.info(f"Model loaded successfully: {state.model.model_name}")
        
        # Initialize chatbot
        if state.vector_db:
            state.chatbot = MultimodalChatbot(
                vector_db=state.vector_db,
                model=state.model,
                use_dspy=False
            )
        
        # Initialize ingestion with vector_db
        state.ingestion = DocumentIngestion()
        state.ingestion.vector_db = state.vector_db  # Add reference to vector_db
        
        logger.info("Application initialized successfully!")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "name": "Multimodal RAG Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "chat": "/api/chat",
            "upload": "/api/documents/upload",
            "search": "/api/search",
            "documents": "/api/documents",
            "health": "/api/health"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vector_db": state.vector_db is not None,
        "chatbot": state.chatbot is not None,
        "model_loaded": state.model is not None
    }

# Import and include routers
from backend.routes import chat, documents, search, model

app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(model.router)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
