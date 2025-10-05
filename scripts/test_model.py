#!/usr/bin/env python3
"""
Test script to verify model fallback behavior.
Run this to check if the model system is working correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.dnotitia_model import DNotitiaModel


def test_model_fallback():
    """Test model loading with fallback."""
    print("=" * 60)
    print("TESTING MODEL FALLBACK SYSTEM")
    print("=" * 60)
    print()
    
    print("📋 Available Models:")
    for i, model_config in enumerate(DNotitiaModel.MODEL_CONFIGS, 1):
        status = "🔒 Gated" if model_config["requires_approval"] else "✅ Open"
        print(f"{i}. {model_config['name']}")
        print(f"   Status: {status}")
        print(f"   Size: {model_config['size_gb']} GB")
        print(f"   Description: {model_config['description']}")
        print()
    
    print("-" * 60)
    print("🔄 Testing Model Load (with fallback enabled)...")
    print("-" * 60)
    print()
    
    try:
        # Try to load model with fallback
        print("Attempting to load primary model...")
        model = DNotitiaModel(
            model_name="dnotitia/DNA-2.0-1.7B",
            use_fallback=True,
            hf_token=None  # No token provided
        )
        
        print()
        print("=" * 60)
        print("✅ MODEL LOADED SUCCESSFULLY!")
        print("=" * 60)
        print()
        
        # Get model info
        info = model.get_model_info()
        
        print("📊 Model Information:")
        print(f"   Requested Model: {info['requested_model']}")
        print(f"   Loaded Model: {info['model_name']}")
        print(f"   Device: {info['device']}")
        print(f"   Description: {info.get('description', 'N/A')}")
        print(f"   Is Fallback: {info.get('is_fallback', False)}")
        print()
        
        if info.get('is_fallback'):
            print("ℹ️  NOTE: Primary model was not accessible.")
            print("    The system automatically fell back to an open model.")
            print("    This is expected behavior if you haven't requested")
            print("    access to the dnotitia model yet.")
        else:
            print("✅ Primary model loaded successfully!")
            print("   You have access to the dnotitia model.")
        
        print()
        print("-" * 60)
        print("🧪 Testing Generation...")
        print("-" * 60)
        print()
        
        test_prompt = "What is artificial intelligence?"
        print(f"Prompt: {test_prompt}")
        print()
        print("Generating response...")
        
        response = model.generate(
            prompt=test_prompt,
            max_new_tokens=100,
            temperature=0.7
        )
        
        print()
        print("Response:")
        print(response[:200] + "..." if len(response) > 200 else response)
        print()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print()
        print("Your chatbot is ready to use! 🎉")
        print()
        
    except Exception as e:
        print()
        print("=" * 60)
        print("❌ ERROR OCCURRED")
        print("=" * 60)
        print()
        print(f"Error: {str(e)}")
        print()
        print("💡 Troubleshooting:")
        print("1. Check your internet connection")
        print("2. Ensure you have enough disk space")
        print("3. Try running with a specific model:")
        print("   python test_model.py --model upstage/SOLAR-10.7B-v1.0")
        print()
        sys.exit(1)


if __name__ == "__main__":
    test_model_fallback()
