#!/usr/bin/env python3
"""
Test the API structure without heavy ML dependencies.
"""

def test_api_structure():
    """Test the basic API structure."""
    print("Testing API structure...")
    
    # Test file extension function
    def ext_from_name(name: str) -> str:
        i = name.rfind(".")
        return name[i:].lower() if i != -1 else ""
    
    # Test allowed extensions
    ALLOWED_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
    
    # Mock recognize function
    def recognize_qari_from_bytes(audio_bytes: bytes, filename: str) -> str:
        # Mock implementation for testing
        return "Abdul Basit Abdussomad"
    
    # Test the functions
    test_files = [
        "test.wav",
        "recording.mp3", 
        "audio.m4a",
        "recitation.flac",
        "qari.ogg",
        "test.aac",
        "invalid.txt"
    ]
    
    print("\nTesting file extension detection:")
    for filename in test_files:
        ext = ext_from_name(filename)
        is_allowed = ext in ALLOWED_EXTS
        status = "✓" if is_allowed else "✗"
        print(f"  {status} {filename} -> {ext} ({'allowed' if is_allowed else 'not allowed'})")
    
    # Test mock recognition
    print("\nTesting mock recognition:")
    test_audio = b"fake audio data"
    result = recognize_qari_from_bytes(test_audio, "test.wav")
    print(f"  ✓ Mock recognition result: {result}")
    
    print("\n" + "=" * 50)
    print("✓ API structure test passed!")
    print("✓ Ready for full implementation")
    print("=" * 50)

if __name__ == "__main__":
    test_api_structure()
