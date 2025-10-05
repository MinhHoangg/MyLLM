"""
Basic tests for the chatbot application.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from app.vector_db_clip import MultimodalVectorDatabase
from app.ingestion import DocumentIngestion
from langchain.schema import Document


def test_vector_db_initialization():
    """Test vector database initialization."""
    db = MultimodalVectorDatabase(persist_directory="data/test_db", use_clip=True)
    assert db is not None
    assert db.collection is not None
    
    # Clean up
    db.clear_collection()


def test_text_embedding():
    """Test text embedding generation."""
    db = MultimodalVectorDatabase(persist_directory="data/test_db", use_clip=True)
    
    texts = ["This is a test.", "Another test sentence."]
    embeddings = db.generate_text_embeddings(texts)
    
    assert len(embeddings) == 2
    assert len(embeddings[0]) > 0
    
    # Clean up
    db.clear_collection()


def test_document_ingestion():
    """Test document ingestion initialization."""
    ingestion = DocumentIngestion()
    
    assert ingestion is not None
    assert len(ingestion.handlers) > 0
    
    supported_exts = ingestion.get_supported_extensions()
    assert '.pdf' in supported_exts
    assert '.txt' in supported_exts


def test_text_chunking():
    """Test text chunking with LangChain."""
    ingestion = DocumentIngestion(chunk_size=50, chunk_overlap=10)
    
    test_data = {
        'text_content': [
            {'text': 'This is a test document. ' * 20}
        ],
        'images': [],
        'metadata': {'file_name': 'test.txt', 'file_type': 'text'}
    }
    
    chunks = ingestion.chunk_documents(test_data)
    
    assert len(chunks) > 0
    assert isinstance(chunks[0], Document)


def test_add_and_search_documents():
    """Test adding documents and searching."""
    db = MultimodalVectorDatabase(persist_directory="data/test_db", use_clip=True)
    
    # Create test documents
    docs = [
        Document(
            page_content="Python is a programming language.",
            metadata={'file_name': 'test1.txt', 'content_type': 'text'}
        ),
        Document(
            page_content="Machine learning is a subset of AI.",
            metadata={'file_name': 'test2.txt', 'content_type': 'text'}
        )
    ]
    
    # Add documents
    ids = db.add_text_documents(docs)
    assert len(ids) == 2
    
    # Search
    results = db.search("programming language", n_results=1)
    assert len(results['documents']) > 0
    
    # Clean up
    db.clear_collection()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
