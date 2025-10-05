# CLIP Integration Guide

## Overview

This chatbot now includes **CLIP (Contrastive Language-Image Pre-training)** integration for true multimodal understanding. CLIP enables the system to understand the semantic relationship between text and images in a unified embedding space.

## What is CLIP?

CLIP is a neural network trained by OpenAI that learns visual concepts from natural language supervision. Unlike traditional approaches that use text descriptions for images, CLIP creates aligned embeddings where:

- **Text and images share the same vector space**
- **Semantically similar text and images are close together**
- **You can search images using text and vice versa**

### Key Benefits

✅ **True Multimodal Search**: Find images using text queries or find text using image queries  
✅ **Better Image Understanding**: Images are represented by their visual content, not just descriptions  
✅ **Cross-Modal Retrieval**: "Show me images similar to this text" or "Find text related to this image"  
✅ **Zero-Shot Classification**: Understand new concepts without additional training  
✅ **Semantic Similarity**: More accurate matching between user intent and content

## Architecture

### Components

1. **CLIPEmbedder** (`models/clip_embedder.py`)
   - Core CLIP encoding functionality
   - Supports text and image encoding
   - Computes similarity between modalities

2. **HybridEmbedder** (`models/clip_embedder.py`)
   - Combines CLIP with sentence-transformers
   - Provides both visual and semantic text embeddings
   - Optional unified embedding space

3. **MultimodalVectorDatabase** (`app/vector_db_clip.py`)
   - Extended ChromaDB with CLIP support
   - Handles text, image, and multimodal queries
   - Late fusion search for combined queries

### Embedding Flow

```
Text Input → CLIP Text Encoder → 512D Vector
                                      ↓
                                Shared Space
                                      ↑
Image Input → CLIP Image Encoder → 512D Vector
```

## Usage

### 1. Basic Setup

```python
from models.clip_embedder import CLIPEmbedder
from app.vector_db_clip import MultimodalVectorDatabase

# Initialize CLIP embedder
embedder = CLIPEmbedder(model_name="openai/clip-vit-base-patch32")

# Initialize multimodal database
db = MultimodalVectorDatabase(
    collection_name="my_collection",
    use_clip=True
)
```

### 2. Adding Documents

#### Text Documents
```python
text_docs = [
    {
        "content": "A beautiful sunset over the ocean",
        "metadata": {"type": "description", "topic": "nature"}
    }
]
db.add_documents(text_docs)
```

#### Image Documents
```python
from PIL import Image

image = Image.open("sunset.jpg")
image_docs = [
    {
        "image": image,
        "metadata": {
            "name": "sunset_photo",
            "description": "Ocean sunset photograph"
        }
    }
]
db.add_image_documents(image_docs)
```

### 3. Searching

#### Text Query (finds both text and images)
```python
results = db.search(
    query="beautiful sunset over water",
    n_results=5
)
```

#### Image Query (finds similar images and related text)
```python
from PIL import Image

query_image = Image.open("query.jpg")
results = db.search(
    query=query_image,
    n_results=5,
    query_type='image'
)
```

#### Multimodal Query (combines text and image)
```python
results = db.search_multimodal(
    text_query="sunset over ocean",
    image_query=query_image,
    n_results=5,
    alpha=0.5  # 0.0 = text only, 1.0 = image only
)
```

### 4. Direct Embedding Operations

```python
from PIL import Image

# Encode text
texts = ["a cat", "a dog", "a bird"]
text_embeddings = embedder.encode_text(texts, normalize=True)

# Encode images
images = [Image.open(f"image{i}.jpg") for i in range(3)]
image_embeddings = embedder.encode_images(images, normalize=True)

# Compute similarity
similarity = embedder.compute_similarity(text_embeddings, image_embeddings)

# Find best matches
matches = embedder.find_best_matches(
    query_texts=["a cute cat"],
    candidate_images=images,
    top_k=3
)
```

## Model Options

### Available CLIP Models

| Model | Size | Embedding Dim | Performance | Use Case |
|-------|------|---------------|-------------|----------|
| `openai/clip-vit-base-patch32` | 600MB | 512 | Balanced | General use (recommended) |
| `openai/clip-vit-large-patch14` | 1.7GB | 768 | High | Maximum accuracy |

### Choosing a Model

```python
# Faster, smaller (default)
embedder = CLIPEmbedder(model_name="openai/clip-vit-base-patch32")

# More accurate, larger
embedder = CLIPEmbedder(model_name="openai/clip-vit-large-patch14")
```

## Integration with Existing Chatbot

### Update Vector Database

Replace the standard `VectorDatabase` with `MultimodalVectorDatabase`:

```python
# Old approach
from app.vector_db import VectorDatabase
db = VectorDatabase(collection_name="docs")

# New approach with CLIP
from app.vector_db_clip import MultimodalVectorDatabase
db = MultimodalVectorDatabase(
    collection_name="docs",
    use_clip=True  # Enable CLIP
)
```

### Processing Images

```python
from PIL import Image
import os

# Load images from directory
image_dir = "data/images"
for filename in os.listdir(image_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path)
        
        db.add_image_documents([{
            "image": image,
            "metadata": {
                "filename": filename,
                "source": "image_collection"
            }
        }])
```

### Enhanced Retrieval

```python
# User asks: "Show me images of red flowers"
results = db.search(
    query="red flowers",
    n_results=5
)

# Results include both:
# 1. Text documents mentioning red flowers
# 2. Images containing red flowers (via CLIP visual understanding)
```

## Performance Optimization

### Batch Processing

```python
# Process images in batches for efficiency
images = [Image.open(f"img{i}.jpg") for i in range(100)]

# Batch size of 16 (default)
embeddings = embedder.encode_images(images, batch_size=16)
```

### Device Selection

```python
# Automatic device selection (CUDA > MPS > CPU)
embedder = CLIPEmbedder()  # Auto-detects best device

# Force specific device
embedder = CLIPEmbedder()
embedder.device = "cpu"  # Force CPU
```

### Caching Embeddings

```python
# Generate embeddings once and store
import numpy as np

image_embeddings = embedder.encode_images(images)
np.save("embeddings.npy", image_embeddings)

# Load later
cached_embeddings = np.load("embeddings.npy")
```

## Advanced Features

### Hybrid Embeddings

Use different models for text and images:

```python
from models.clip_embedder import HybridEmbedder

hybrid = HybridEmbedder(
    clip_model_name="openai/clip-vit-base-patch32",
    text_model_name="sentence-transformers/all-MiniLM-L6-v2",
    use_unified_space=True  # Project to same dimension
)

# Use CLIP for images, sentence-transformers for text
text_emb = hybrid.encode_text(["query"], use_clip=False)
image_emb = hybrid.encode_images([image])
```

### Custom Similarity Thresholds

```python
# Filter results by similarity score
results = db.search("query", n_results=10)
filtered = [r for r in results if r['score'] > 0.7]
```

### Multimodal Fusion

```python
# Adjust text vs image importance
results = db.search_multimodal(
    text_query="sunset",
    image_query=example_image,
    alpha=0.7  # 70% text, 30% image
)
```

## Testing

Run the comprehensive test suite:

```bash
python scripts/test_clip.py
```

This tests:
- ✅ CLIPEmbedder initialization
- ✅ Text and image encoding
- ✅ Similarity computation
- ✅ HybridEmbedder functionality
- ✅ MultimodalVectorDatabase operations
- ✅ Text, image, and multimodal search

## Troubleshooting

### Issue: Out of Memory

**Solution**: Use smaller batch sizes or smaller model

```python
# Reduce batch size
embeddings = embedder.encode_images(images, batch_size=4)

# Or use smaller model
embedder = CLIPEmbedder(model_name="openai/clip-vit-base-patch32")
```

### Issue: Slow Performance

**Solution**: Ensure you're using GPU/MPS

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

# Force device
embedder = CLIPEmbedder()
embedder.to("mps")  # or "cuda" or "cpu"
```

### Issue: Poor Image Matching

**Solution**: 
1. Use larger CLIP model (ViT-L/14)
2. Ensure images are preprocessed correctly
3. Normalize embeddings

```python
embedder = CLIPEmbedder(model_name="openai/clip-vit-large-patch14")
embeddings = embedder.encode_images(images, normalize=True)
```

### Issue: CLIP Model Download Fails

**Solution**: Pre-download models

```bash
python -c "from transformers import CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"
```

## Comparison: Before vs After CLIP

### Before (Text-Only)

```python
# Image stored as text description
doc = {
    "content": "Image of a red sunset over the ocean",
    "metadata": {"type": "image_description"}
}
db.add_documents([doc])

# Search finds description, not visual content
results = db.search("orange sky at dusk")
# May miss the image if description doesn't mention "orange"
```

### After (With CLIP)

```python
# Image stored with visual embeddings
image = Image.open("sunset.jpg")
doc = {
    "image": image,
    "metadata": {"filename": "sunset.jpg"}
}
db.add_image_documents([doc])

# Search understands visual content
results = db.search("orange sky at dusk")
# Finds the sunset image because CLIP understands
# that sunsets have orange/red skies
```

## Best Practices

1. **Use Normalized Embeddings**: Always normalize for better similarity comparison
   ```python
   embeddings = embedder.encode_text(texts, normalize=True)
   ```

2. **Batch Process**: Process multiple images at once for efficiency
   ```python
   embeddings = embedder.encode_images(images, batch_size=16)
   ```

3. **Store Metadata**: Include rich metadata for filtering and context
   ```python
   doc = {
       "image": image,
       "metadata": {
           "filename": "photo.jpg",
           "timestamp": "2024-01-01",
           "category": "nature",
           "tags": ["sunset", "ocean"]
       }
   }
   ```

4. **Choose Appropriate Model**: Use ViT-B/32 for speed, ViT-L/14 for accuracy

5. **Test Thoroughly**: Use `test_clip.py` to validate your setup

## References

- [CLIP Paper](https://arxiv.org/abs/2103.00020) - Learning Transferable Visual Models From Natural Language Supervision
- [OpenAI CLIP](https://github.com/openai/CLIP) - Official CLIP repository
- [Hugging Face CLIP](https://huggingface.co/docs/transformers/model_doc/clip) - CLIP documentation

## Next Steps

1. ✅ Install CLIP dependencies: `pip install -r requirements.txt`
2. ✅ Run test suite: `python scripts/test_clip.py`
3. 🔄 Update your application to use `MultimodalVectorDatabase`
4. 🔄 Process your image collection with CLIP embeddings
5. 🔄 Test multimodal search in your chatbot

---

**Need Help?** Check the test script examples or refer to the API documentation in the source files.
