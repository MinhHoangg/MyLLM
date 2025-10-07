"""
CLIP Model Wrapper: Enhanced multimodal embeddings using OpenAI's CLIP.
Provides unified embedding space for both text and images.
"""

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from typing import List, Union, Optional, Dict, Any
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CLIPEmbedder:
    """
    CLIP-based embedder for multimodal content.
    Creates aligned embeddings for both text and images.
    """
    
    # Available CLIP models
    AVAILABLE_MODELS = [
        {
            "name": "openai/clip-vit-base-patch32",
            "description": "CLIP ViT-B/32 (fast, good quality)",
            "size_mb": 600,
            "embedding_dim": 512
        },
        {
            "name": "openai/clip-vit-large-patch14",
            "description": "CLIP ViT-L/14 (slower, best quality)",
            "size_mb": 1700,
            "embedding_dim": 768
        }
    ]
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None
    ):
        """
        Initialize CLIP model for multimodal embeddings.
        
        Args:
            model_name: CLIP model identifier from Hugging Face
            device: Device to run on ('cuda', 'mps', 'cpu', or None for auto)
        """
        self.model_name = model_name
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading CLIP model: {model_name} on {self.device}")
        
        # Load CLIP model and processor
        try:
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            
            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Get embedding dimension
            self.embedding_dim = self.model.config.projection_dim
            
            logger.info(f" CLIP model loaded successfully")
            logger.info(f" Embedding dimension: {self.embedding_dim}")
            logger.info(f" Device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {str(e)}")
            raise
    
    def encode_text(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Encode text into embeddings using CLIP.
        
        Args:
            texts: Single text string or list of texts
            normalize: Whether to normalize embeddings (recommended)
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings (shape: [n_texts, embedding_dim])
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                
                # Normalize if requested
                if normalize:
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                embeddings.append(text_features.cpu().numpy())
        
        # Concatenate all batches
        embeddings = np.vstack(embeddings)
        
        return embeddings
    
    def encode_images(
        self,
        images: Union[Image.Image, List[Image.Image], str, List[str]],
        normalize: bool = True,
        batch_size: int = 16
    ) -> np.ndarray:
        """
        Encode images into embeddings using CLIP.
        
        Args:
            images: Single image, list of images, or paths to images
            normalize: Whether to normalize embeddings (recommended)
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings (shape: [n_images, embedding_dim])
        """
        # Load images if paths provided
        if isinstance(images, str):
            images = [Image.open(images).convert("RGB")]
        elif isinstance(images, list) and isinstance(images[0], str):
            images = [Image.open(img_path).convert("RGB") for img_path in images]
        elif isinstance(images, Image.Image):
            images = [images]
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Preprocess images
            inputs = self.processor(
                images=batch_images,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                
                # Normalize if requested
                if normalize:
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                embeddings.append(image_features.cpu().numpy())
        
        # Concatenate all batches
        embeddings = np.vstack(embeddings)
        
        return embeddings
    
    def compute_similarity(
        self,
        text_embeddings: np.ndarray,
        image_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarity between text and image embeddings.
        
        Args:
            text_embeddings: Text embeddings (shape: [n_texts, embedding_dim])
            image_embeddings: Image embeddings (shape: [n_images, embedding_dim])
            
        Returns:
            Similarity matrix (shape: [n_texts, n_images])
        """
        # Cosine similarity (assumes normalized embeddings)
        similarity = np.matmul(text_embeddings, image_embeddings.T)
        return similarity
    
    def find_best_matches(
        self,
        query: Union[str, Image.Image],
        candidates: Union[List[str], List[Image.Image]],
        top_k: int = 5,
        query_type: str = "auto"
    ) -> List[Dict[str, Any]]:
        """
        Find best matches for a query among candidates.
        
        Args:
            query: Query text or image
            candidates: List of candidate texts or images
            top_k: Number of top matches to return
            query_type: "text", "image", or "auto" (auto-detect)
            
        Returns:
            List of dicts with 'index', 'score', and 'candidate'
        """
        # Auto-detect query type
        if query_type == "auto":
            if isinstance(query, str):
                query_type = "text"
            elif isinstance(query, Image.Image):
                query_type = "image"
            else:
                raise ValueError("Cannot auto-detect query type")
        
        # Encode query
        if query_type == "text":
            query_embedding = self.encode_text(query)
        else:
            query_embedding = self.encode_images(query)
        
        # Encode candidates
        if isinstance(candidates[0], str):
            candidate_embeddings = self.encode_text(candidates)
        else:
            candidate_embeddings = self.encode_images(candidates)
        
        # Compute similarities
        similarities = self.compute_similarity(query_embedding, candidate_embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Format results
        results = []
        for idx in top_indices:
            results.append({
                'index': int(idx),
                'score': float(similarities[idx]),
                'candidate': candidates[idx]
            })
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the CLIP model.
        
        Returns:
            Dictionary with model details
        """
        model_config = next(
            (m for m in self.AVAILABLE_MODELS if m["name"] == self.model_name),
            None
        )
        
        info = {
            'model_name': self.model_name,
            'device': self.device,
            'embedding_dim': self.embedding_dim,
            'max_text_length': 77
        }
        
        if model_config:
            info['description'] = model_config['description']
            info['size_mb'] = model_config['size_mb']
        
        return info
    
    @classmethod
    def list_available_models(cls) -> List[Dict[str, Any]]:
        """
        Get list of available CLIP models.
        
        Returns:
            List of model configurations
        """
        return cls.AVAILABLE_MODELS.copy()


class HybridEmbedder:
    """
    Hybrid embedder combining CLIP (for multimodal) and sentence-transformers (for text).
    Uses CLIP for image-text alignment and sentence-transformers for pure text tasks.
    """
    
    def __init__(
        self,
        clip_model: str = "openai/clip-vit-base-patch32",
        text_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        use_clip_for_text: bool = False
    ):
        """
        Initialize hybrid embedder.
        
        Args:
            clip_model: CLIP model name for images and cross-modal tasks
            text_model: Sentence-transformer model for text-only tasks
            device: Device to use
            use_clip_for_text: If True, use CLIP for all text (unified space)
        """
        from sentence_transformers import SentenceTransformer
        
        self.use_clip_for_text = use_clip_for_text
        
        # Initialize CLIP
        self.clip_embedder = CLIPEmbedder(model_name=clip_model, device=device)
        
        # Initialize sentence-transformer (optional)
        if not use_clip_for_text:
            logger.info(f"Loading sentence-transformer: {text_model}")
            self.text_embedder = SentenceTransformer(text_model)
            if device:
                self.text_embedder = self.text_embedder.to(device)
        else:
            self.text_embedder = None
        
        logger.info(f" Hybrid embedder initialized")
    
    def encode_text(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True
    ) -> np.ndarray:
        """Encode text using appropriate model."""
        if self.use_clip_for_text:
            return self.clip_embedder.encode_text(texts, normalize=normalize)
        else:
            if isinstance(texts, str):
                texts = [texts]
            embeddings = self.text_embedder.encode(texts, show_progress_bar=False)
            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / norms
            return embeddings
    
    def encode_images(
        self,
        images: Union[Image.Image, List[Image.Image], str, List[str]],
        normalize: bool = True
    ) -> np.ndarray:
        """Encode images using CLIP."""
        return self.clip_embedder.encode_images(images, normalize=normalize)
    
    def get_embedding_dim(self, modality: str = "text") -> int:
        """Get embedding dimension for specified modality."""
        if modality == "image" or self.use_clip_for_text:
            return self.clip_embedder.embedding_dim
        else:
            return self.text_embedder.get_sentence_embedding_dimension()
