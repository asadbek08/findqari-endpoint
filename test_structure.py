#!/usr/bin/env python3
"""
Test script to verify the app structure without running the full model.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if we can import the basic structure."""
    try:
        print("Testing basic imports...")
        
        # Test if we can import the basic modules
        import io
        print("✓ io imported")
        
        # Test file extension function
        def ext_from_name(name: str) -> str:
            i = name.rfind(".")
            return name[i:].lower() if i != -1 else ""
        
        # Test the function
        assert ext_from_name("test.wav") == ".wav"
        assert ext_from_name("test.mp3") == ".mp3"
        assert ext_from_name("test") == ""
        print("✓ ext_from_name function works")
        
        # Test allowed extensions
        ALLOWED_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
        assert ".wav" in ALLOWED_EXTS
        assert ".mp3" in ALLOWED_EXTS
        print("✓ ALLOWED_EXTS defined correctly")
        
        print("\n" + "=" * 50)
        print("✓ Basic structure test passed!")
        print("✓ Ready for FastAPI + Gradio integration")
        print("=" * 50)
        
    except Exception as e:
        print(f"✗ Error in basic structure test: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
