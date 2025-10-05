"""
Enhanced Vector Database with CLIP support for true multimodal embeddings.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
from langchain.schema import Document
import uuid
import logging

logger = logging.getLogger(__name__)


class MultimodalVectorDatabase:
    """
    Enhanced vector database with CLIP support for multimodal embeddings.
    Handles both text and images in a unified embedding space.
    """
    
    def __init__(
        self,
        persist_directory: str = "data/embeddings_db",
        collection_name: str = "multimodal_docs",
        use_clip: bool = True,
        clip_model: str = "openai/clip-vit-base-patch32",
        text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize multimodal vector database.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection
            use_clip: Whether to use CLIP for multimodal embeddings
            clip_model: CLIP model name (if use_clip=True)
            text_model: Fallback text embedding model
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.use_clip = use_clip
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection_name = collection_name
        
        # Initialize embedding models
        if use_clip:
            logger.info("🎨 Initializing CLIP-based multimodal embeddings")
            from models.clip_embedder import HybridEmbedder
            
            self.embedder = HybridEmbedder(
                clip_model=clip_model,
                text_model=text_model,
                use_clip_for_text=True  # Unified embedding space
            )
            self.embedding_dim = self.embedder.clip_embedder.embedding_dim
        else:
            logger.info("📝 Using text-only embeddings (sentence-transformers)")
            self.embedder = SentenceTransformer(text_model)
            self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        
        # Get or create collections
        self.collection = self._get_or_create_collection()
        
        logger.info(f"✅ Vector database initialized")
        logger.info(f"   CLIP enabled: {use_clip}")
        logger.info(f"   Embedding dimension: {self.embedding_dim}")
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one."""
        try:
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
        
        return collection
    
    def generate_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for text.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        if self.use_clip:
            embeddings = self.embedder.encode_text(texts, normalize=True)
        else:
            embeddings = self.embedder.encode(texts, show_progress_bar=False)
        
        return embeddings.tolist()
    
    def generate_image_embeddings(
        self,
        images: Union[List[Image.Image], List[str]]
    ) -> List[List[float]]:
        """
        Generate embeddings for images using CLIP.
        
        Args:
            images: List of PIL Images or paths to images
            
        Returns:
            List of embedding vectors
        """
        if not self.use_clip:
            raise ValueError("CLIP must be enabled to embed images")
        
        embeddings = self.embedder.encode_images(images, normalize=True)
        return embeddings.tolist()
    
    def add_text_documents(self, documents: List[Document]) -> List[str]:
        """
        Add text documents to the vector database.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        # Extract texts and metadata
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Generate embeddings
        embeddings = self.generate_text_embeddings(texts)
        
        # Generate unique IDs
        ids = [str(uuid.uuid4()) for _ in documents]
        
        # Process metadata
        processed_metadatas = []
        for metadata in metadatas:
            processed_metadata = {}
            for key, value in metadata.items():
                processed_metadata[key] = str(value)
            # Mark as text document
            processed_metadata['content_type'] = 'text'
            processed_metadatas.append(processed_metadata)
        
        # Add to collection
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=processed_metadatas,
            ids=ids
        )
        
        logger.info(f"✅ Added {len(documents)} text documents")
        return ids
    
    def add_image_documents(
        self,
        image_documents: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Add image documents with CLIP embeddings.
        
        Args:
            image_documents: List of dicts with 'image' (PIL.Image or path)
                           and 'metadata' keys
            
        Returns:
            List of document IDs
        """
        if not image_documents:
            return []
        
        if not self.use_clip:
            # Fallback: use text descriptions
            return self._add_image_documents_text_only(image_documents)
        
        # Extract images and metadata
        images = []
        metadatas = []
        descriptions = []
        
        for img_doc in image_documents:
            # Get image
            if 'image' in img_doc:
                img = img_doc['image']
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")
                images.append(img)
            elif 'image_path' in img_doc:
                img = Image.open(img_doc['image_path']).convert("RGB")
                images.append(img)
            else:
                raise ValueError("Image document must have 'image' or 'image_path'")
            
            # Get metadata
            metadata = img_doc.get('metadata', {})
            
            # Create description
            image_id = metadata.get('image_id', 'unknown')
            description_parts = [f"Image: {image_id}"]
            
            if 'caption' in metadata:
                description_parts.append(f"Caption: {metadata['caption']}")
            if 'format' in metadata:
                description_parts.append(f"Format: {metadata['format']}")
            if 'file_name' in metadata:
                description_parts.append(f"From: {metadata['file_name']}")
            
            description = " | ".join(description_parts)
            descriptions.append(description)
            
            # Process metadata
            processed_metadata = {}
            for key, value in metadata.items():
                processed_metadata[key] = str(value)
            processed_metadata['content_type'] = 'image'
            metadatas.append(processed_metadata)
        
        # Generate CLIP embeddings for images
        embeddings = self.generate_image_embeddings(images)
        
        # Generate unique IDs
        ids = [str(uuid.uuid4()) for _ in image_documents]
        
        # Add to collection (using descriptions as documents)
        self.collection.add(
            documents=descriptions,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"✅ Added {len(image_documents)} image documents with CLIP embeddings")
        return ids
    
    def _add_image_documents_text_only(
        self,
        image_documents: List[Dict[str, Any]]
    ) -> List[str]:
        """Fallback: add images using text descriptions only."""
        texts = []
        metadatas = []
        
        for img_doc in image_documents:
            metadata = img_doc.get('metadata', {})
            image_id = metadata.get('image_id', 'unknown')
            
            # Create descriptive text
            description_parts = [
                f"Image: {image_id}",
                f"Format: {metadata.get('format', 'unknown')}",
                f"Size: {metadata.get('dimensions', 'unknown')}"
            ]
            
            if 'page' in metadata:
                description_parts.append(f"Page: {metadata['page']}")
            if 'file_name' in metadata:
                description_parts.append(f"From: {metadata['file_name']}")
            if 'caption' in metadata:
                description_parts.append(f"Caption: {metadata['caption']}")
            
            text = " | ".join(description_parts)
            texts.append(text)
            
            # Process metadata
            processed_metadata = {}
            for key, value in metadata.items():
                processed_metadata[key] = str(value)
            processed_metadata['content_type'] = 'image'
            metadatas.append(processed_metadata)
        
        # Generate text embeddings
        embeddings = self.generate_text_embeddings(texts)
        
        # Generate IDs
        ids = [str(uuid.uuid4()) for _ in image_documents]
        
        # Add to collection
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Added {len(image_documents)} images (text-based)")
        return ids
    
    def search(
        self,
        query: Union[str, Image.Image],
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search for similar documents using text or image query.
        
        Args:
            query: Search query (text string or PIL Image)
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            content_type: Filter by 'text', 'image', or None (all)
            
        Returns:
            Dictionary containing search results
        """
        # Generate query embedding
        if isinstance(query, str):
            query_embedding = self.generate_text_embeddings([query])[0]
        elif isinstance(query, Image.Image):
            if not self.use_clip:
                raise ValueError("Image queries require CLIP to be enabled")
            query_embedding = self.generate_image_embeddings([query])[0]
        else:
            raise ValueError("Query must be string or PIL Image")
        
        # Add content type filter
        where_filter = filter_metadata or {}
        if content_type:
            where_filter['content_type'] = content_type
        
        # Perform search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter if where_filter else None
        )
        
        return {
            'documents': results['documents'][0] if results['documents'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else [],
            'distances': results['distances'][0] if results['distances'] else [],
            'ids': results['ids'][0] if results['ids'] else []
        }
    
    def search_multimodal(
        self,
        text_query: Optional[str] = None,
        image_query: Optional[Image.Image] = None,
        n_results: int = 5,
        alpha: float = 0.5
    ) -> Dict[str, Any]:
        """
        Search using both text and image queries (late fusion).
        
        Args:
            text_query: Text query string
            image_query: PIL Image query
            n_results: Number of results
            alpha: Weight for text (1-alpha for image)
            
        Returns:
            Dictionary with combined search results
        """
        if not text_query and not image_query:
            raise ValueError("At least one query (text or image) must be provided")
        
        if text_query and not image_query:
            return self.search(text_query, n_results=n_results)
        
        if image_query and not text_query:
            return self.search(image_query, n_results=n_results)
        
        # Both queries provided - combine results
        text_results = self.search(text_query, n_results=n_results * 2)
        image_results = self.search(image_query, n_results=n_results * 2)
        
        # Combine scores with alpha weighting
        combined_scores = {}
        
        for doc_id, distance in zip(text_results['ids'], text_results['distances']):
            combined_scores[doc_id] = {'score': alpha * (1 - distance), 'from': 'text'}
        
        for doc_id, distance in zip(image_results['ids'], image_results['distances']):
            score = (1 - alpha) * (1 - distance)
            if doc_id in combined_scores:
                combined_scores[doc_id]['score'] += score
                combined_scores[doc_id]['from'] = 'both'
            else:
                combined_scores[doc_id] = {'score': score, 'from': 'image'}
        
        # Sort by combined score
        sorted_ids = sorted(
            combined_scores.keys(),
            key=lambda x: combined_scores[x]['score'],
            reverse=True
        )[:n_results]
        
        # Fetch full details
        all_docs = self.collection.get(ids=sorted_ids)
        
        # Reorder results
        id_to_idx = {doc_id: idx for idx, doc_id in enumerate(all_docs['ids'])}
        
        results = {
            'documents': [all_docs['documents'][id_to_idx[doc_id]] for doc_id in sorted_ids],
            'metadatas': [all_docs['metadatas'][id_to_idx[doc_id]] for doc_id in sorted_ids],
            'scores': [combined_scores[doc_id]['score'] for doc_id in sorted_ids],
            'sources': [combined_scores[doc_id]['from'] for doc_id in sorted_ids],
            'ids': sorted_ids
        }
        
        return results
    
    # Inherit other methods from VectorDatabase
    def get_all_documents(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Retrieve all documents."""
        try:
            results = self.collection.get(limit=limit)
            return results
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return {'documents': [], 'metadatas': [], 'ids': []}
    
    def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents by IDs."""
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return False
    
    def clear_collection(self) -> bool:
        """Clear all documents."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self._get_or_create_collection()
            logger.info(f"Cleared collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            count = self.collection.count()
            return {
                'name': self.collection_name,
                'count': count,
                'persist_directory': str(self.persist_directory),
                'clip_enabled': self.use_clip,
                'embedding_dim': self.embedding_dim
            }
        except Exception as e:
            return {
                'name': self.collection_name,
                'count': 0,
                'error': str(e)
            }
