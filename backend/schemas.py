"""
Pydantic schemas for API request/response validation
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# Chat schemas
class ChatRequest(BaseModel):
    question: str = Field(..., description="User's question")
    use_rag: Optional[bool] = Field(None, description="Whether to use RAG (None=auto-detect)")
    include_history: bool = Field(True, description="Include conversation history")
    similarity_threshold: float = Field(0.3, description="Minimum similarity score for retrieval")

class ChatResponse(BaseModel):
    question: str
    answer: str
    context: str
    sources: List[Dict[str, Any]]
    method: str  # 'rag', 'direct', or 'cached'
    used_rag: bool

# Document Upload schemas
class UploadResponse(BaseModel):
    status: str
    files_processed: int
    chunks: int  # Total text chunks
    images: int  # Total images
    results: List[Dict[str, Any]]

class DocumentMetadata(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    similarity: Optional[float] = None

# Search schemas
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    content_type: Optional[str] = Field(None, description="Filter by 'text' or 'image'")
    similarity_threshold: float = Field(0.3, description="Minimum similarity score")

class SearchResponse(BaseModel):
    query: str
    documents: List[DocumentMetadata]
    total_results: int
    similarity_threshold: float
    metadata: Dict[str, Any] = {}

# Database schemas
class DatabaseStats(BaseModel):
    name: str
    count: int
    persist_directory: str
    clip_enabled: bool
    embedding_dim: int

class DocumentInfo(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]

# Model Configuration schemas
class ModelConfig(BaseModel):
    high_parameter: bool = Field(..., description="Use high-performance model (7.8B vs 2.4B)")

class ModelInfo(BaseModel):
    model_name: str
    model_type: str
    device: str
    is_loaded: bool
    high_parameter: bool
    max_tokens: int
    parameters: str

# Settings schemas
class Settings(BaseModel):
    use_rag: Optional[bool] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    include_history: bool = True
    similarity_threshold: float = 0.3
