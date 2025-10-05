#!/usr/bin/env python3
"""
Check which model will be loaded and provide status information.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.dnotitia_model import DNotitiaModel
import os


def check_model_status():
    """Check model access status."""
    print("=" * 70)
    print("MODEL STATUS CHECKER")
    print("=" * 70)
    print()
    
    # Check for HF token
    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    if hf_token:
        print("✅ HF Token found in environment")
        print(f"   Token: {hf_token[:10]}...")
    else:
        print("ℹ️  No HF token found in environment")
        print("   Open models will be used")
    
    print()
    print("-" * 70)
    print("Available Models:")
    print("-" * 70)
    print()
    
    for i, config in enumerate(DNotitiaModel.MODEL_CONFIGS, 1):
        print(f"{i}. {config['name']}")
        print(f"   Description: {config['description']}")
        print(f"   Size: {config['size_gb']} GB")
        
        if config['requires_approval']:
            print(f"   Status: 🔒 GATED (requires approval)")
            if hf_token:
                print(f"   → Will try to load (you have token)")
            else:
                print(f"   → Will skip (no token provided)")
        else:
            print(f"   Status: ✅ OPEN ACCESS")
            print(f"   → Available for immediate use")
        
        print()
    
    print("-" * 70)
    print("Expected Behavior:")
    print("-" * 70)
    print()
    
    if hf_token:
        print("With HF token provided:")
        print("  1. Try dnotitia/DNA-2.0-1.7B")
        print("     • If approved: ✅ Use it")
        print("     • If not approved: → Fallback to SOLAR")
        print("  2. Fallback: upstage/SOLAR-10.7B-v1.0 ✅")
    else:
        print("Without HF token:")
        print("  1. Skip dnotitia/DNA-2.0-1.7B (requires auth)")
        print("  2. Use: upstage/SOLAR-10.7B-v1.0 ✅")
    
    print()
    print("-" * 70)
    print("Recommendations:")
    print("-" * 70)
    print()
    
    if not hf_token:
        print("💡 You can use the app immediately with SOLAR model!")
        print()
        print("   To use dnotitia model later:")
        print("   1. Request access: https://huggingface.co/dnotitia/DNA-2.0-1.7B")
        print("   2. Get token: https://huggingface.co/settings/tokens")
        print("   3. Set environment variable:")
        print("      export HUGGING_FACE_HUB_TOKEN='your_token'")
    else:
        print("✅ You're all set!")
        print("   The app will try the primary model first,")
        print("   then fallback to SOLAR if needed.")
    
    print()
    print("=" * 70)
    print("Ready to run: streamlit run ui/app.py")
    print("=" * 70)
    print()


if __name__ == "__main__":
    check_model_status()
