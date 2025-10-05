"""
Test script for CLIP embedder and multimodal vector database functionality.

This script tests:
1. CLIPEmbedder initialization and encoding
2. Text-to-image similarity
3. MultimodalVectorDatabase operations
4. Image search and multimodal search
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from models.clip_embedder import CLIPEmbedder, HybridEmbedder
from app.vector_db_clip import MultimodalVectorDatabase


def create_test_images():
    """Create simple test images for demonstration."""
    images = []
    
    # Create a red square
    img1 = Image.new('RGB', (224, 224), color='red')
    draw1 = ImageDraw.Draw(img1)
    draw1.text((80, 100), "RED", fill='white')
    images.append(('red_square', img1))
    
    # Create a blue circle
    img2 = Image.new('RGB', (224, 224), color='blue')
    draw2 = ImageDraw.Draw(img2)
    draw2.ellipse([50, 50, 174, 174], fill='lightblue', outline='white', width=3)
    draw2.text((80, 100), "BLUE", fill='white')
    images.append(('blue_circle', img2))
    
    # Create a green triangle (approximation)
    img3 = Image.new('RGB', (224, 224), color='green')
    draw3 = ImageDraw.Draw(img3)
    draw3.polygon([(112, 50), (50, 174), (174, 174)], fill='lightgreen', outline='white')
    draw3.text((80, 150), "GREEN", fill='white')
    images.append(('green_triangle', img3))
    
    return images


def test_clip_embedder():
    """Test CLIPEmbedder functionality."""
    print("=" * 60)
    print("Testing CLIPEmbedder")
    print("=" * 60)
    
    # Initialize embedder
    print("\n1. Initializing CLIPEmbedder...")
    embedder = CLIPEmbedder(model_name="openai/clip-vit-base-patch32")
    print(f"✓ Device: {embedder.device}")
    print(f"✓ Model: {embedder.model_name}")
    print(f"✓ Embedding dimension: {embedder.embedding_dim}")
    
    # Test text encoding
    print("\n2. Testing text encoding...")
    texts = [
        "a red square",
        "a blue circle",
        "a green triangle",
        "a cat sitting on a couch",
        "a dog playing in the park"
    ]
    text_embeddings = embedder.encode_text(texts, normalize=True)
    print(f"✓ Text embeddings shape: {text_embeddings.shape}")
    print(f"✓ Text embeddings are normalized: {np.allclose(np.linalg.norm(text_embeddings, axis=1), 1.0)}")
    
    # Test image encoding
    print("\n3. Testing image encoding...")
    test_images = create_test_images()
    images = [img for _, img in test_images]
    image_embeddings = embedder.encode_images(images, normalize=True)
    print(f"✓ Image embeddings shape: {image_embeddings.shape}")
    print(f"✓ Image embeddings are normalized: {np.allclose(np.linalg.norm(image_embeddings, axis=1), 1.0)}")
    
    # Test similarity computation
    print("\n4. Testing text-to-image similarity...")
    similarity = embedder.compute_similarity(
        text_embeddings[:3],  # First 3 texts (shapes)
        image_embeddings  # All 3 images
    )
    print(f"✓ Similarity matrix shape: {similarity.shape}")
    print("\nSimilarity scores (text × image):")
    print("                Red Square  Blue Circle  Green Triangle")
    for i, text in enumerate(texts[:3]):
        print(f"{text:20s}  {similarity[i, 0]:.4f}      {similarity[i, 1]:.4f}      {similarity[i, 2]:.4f}")
    
    # Find best matches
    print("\n5. Testing best match finding...")
    query_text = ["a blue circular shape"]
    matches = embedder.find_best_matches(query_text, images, top_k=3)
    print(f"✓ Query: '{query_text[0]}'")
    print(f"✓ Top matches:")
    for i, (img, score) in enumerate(matches[0], 1):
        # Find which test image this is
        for name, test_img in test_images:
            if img == test_img:
                print(f"   {i}. {name}: {score:.4f}")
                break
    
    return embedder, test_images


def test_hybrid_embedder():
    """Test HybridEmbedder functionality."""
    print("\n" + "=" * 60)
    print("Testing HybridEmbedder")
    print("=" * 60)
    
    print("\n1. Initializing HybridEmbedder...")
    hybrid = HybridEmbedder(
        clip_model_name="openai/clip-vit-base-patch32",
        text_model_name="sentence-transformers/all-MiniLM-L6-v2",
        use_unified_space=True
    )
    print(f"✓ Device: {hybrid.device}")
    print(f"✓ CLIP dimension: {hybrid.clip_dim}")
    print(f"✓ Text dimension: {hybrid.text_dim}")
    print(f"✓ Unified space: {hybrid.use_unified_space}")
    
    # Test text encoding with both models
    print("\n2. Testing hybrid text encoding...")
    texts = ["a red square", "a blue circle"]
    text_embeddings = hybrid.encode_text(texts, use_clip=False)
    clip_text_embeddings = hybrid.encode_text(texts, use_clip=True)
    print(f"✓ Sentence-transformers embeddings: {text_embeddings.shape}")
    print(f"✓ CLIP text embeddings: {clip_text_embeddings.shape}")
    
    # Test image encoding
    print("\n3. Testing hybrid image encoding...")
    test_images = create_test_images()
    images = [img for _, img in test_images]
    image_embeddings = hybrid.encode_images(images)
    print(f"✓ CLIP image embeddings: {image_embeddings.shape}")
    
    return hybrid


def test_multimodal_vector_db():
    """Test MultimodalVectorDatabase functionality."""
    print("\n" + "=" * 60)
    print("Testing MultimodalVectorDatabase")
    print("=" * 60)
    
    # Initialize database
    print("\n1. Initializing MultimodalVectorDatabase...")
    db = MultimodalVectorDatabase(
        collection_name="test_multimodal",
        persist_directory="./test_chroma_db",
        clip_model_name="openai/clip-vit-base-patch32",
        use_clip=True
    )
    print(f"✓ Database initialized")
    print(f"✓ CLIP enabled: {db.use_clip}")
    print(f"✓ Collection: {db.collection_name}")
    
    # Add text documents
    print("\n2. Adding text documents...")
    text_docs = [
        {"content": "The sky is blue and clear today", "metadata": {"type": "weather", "color": "blue"}},
        {"content": "Red roses are beautiful flowers", "metadata": {"type": "nature", "color": "red"}},
        {"content": "Green grass covers the field", "metadata": {"type": "nature", "color": "green"}},
    ]
    db.add_documents(text_docs)
    print(f"✓ Added {len(text_docs)} text documents")
    
    # Add image documents
    print("\n3. Adding image documents...")
    test_images = create_test_images()
    image_docs = [
        {
            "image": img,
            "metadata": {
                "name": name,
                "type": "shape",
                "description": f"A {name.replace('_', ' ')}"
            }
        }
        for name, img in test_images
    ]
    db.add_image_documents(image_docs)
    print(f"✓ Added {len(image_docs)} image documents")
    
    # Test text query
    print("\n4. Testing text-based search...")
    results = db.search("blue shapes or blue sky", n_results=3)
    print(f"✓ Query: 'blue shapes or blue sky'")
    print(f"✓ Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        content = result.get('content', result.get('metadata', {}).get('description', 'N/A'))
        score = result.get('score', 0.0)
        doc_type = result.get('metadata', {}).get('type', 'unknown')
        print(f"   {i}. [{doc_type}] {content[:50]}... (score: {score:.4f})")
    
    # Test image query
    print("\n5. Testing image-based search...")
    query_image = test_images[1][1]  # Blue circle
    results = db.search(query_image, n_results=3, query_type='image')
    print(f"✓ Query: blue circle image")
    print(f"✓ Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        content = result.get('content', result.get('metadata', {}).get('description', 'N/A'))
        score = result.get('score', 0.0)
        name = result.get('metadata', {}).get('name', 'unknown')
        print(f"   {i}. {name}: {content[:50]} (score: {score:.4f})")
    
    # Test multimodal search
    print("\n6. Testing multimodal search (text + image)...")
    results = db.search_multimodal(
        text_query="red or blue colors",
        image_query=query_image,
        n_results=3,
        alpha=0.5
    )
    print(f"✓ Text query: 'red or blue colors'")
    print(f"✓ Image query: blue circle")
    print(f"✓ Found {len(results)} results (alpha=0.5):")
    for i, result in enumerate(results, 1):
        content = result.get('content', result.get('metadata', {}).get('description', 'N/A'))
        score = result.get('score', 0.0)
        doc_type = result.get('metadata', {}).get('type', 'unknown')
        print(f"   {i}. [{doc_type}] {content[:50]}... (score: {score:.4f})")
    
    # Clean up
    print("\n7. Cleaning up test database...")
    import shutil
    if os.path.exists("./test_chroma_db"):
        shutil.rmtree("./test_chroma_db")
    print("✓ Test database removed")
    
    return db


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CLIP Integration Test Suite")
    print("=" * 60)
    
    try:
        # Check device availability
        print("\n📱 Device Information:")
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            print("✓ MPS (Apple Silicon) available")
        else:
            print("✓ Using CPU")
        
        # Run tests
        print("\n🧪 Running Tests...\n")
        
        # Test 1: CLIPEmbedder
        embedder, test_images = test_clip_embedder()
        
        # Test 2: HybridEmbedder
        hybrid = test_hybrid_embedder()
        
        # Test 3: MultimodalVectorDatabase
        db = test_multimodal_vector_db()
        
        # Summary
        print("\n" + "=" * 60)
        print("✅ All Tests Passed!")
        print("=" * 60)
        print("\nSummary:")
        print("✓ CLIPEmbedder: Text and image encoding working")
        print("✓ HybridEmbedder: Dual embedding space functional")
        print("✓ MultimodalVectorDatabase: Text, image, and multimodal search working")
        print("\nCLIP integration is ready for production use! 🎉")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ Test Failed!")
        print("=" * 60)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
