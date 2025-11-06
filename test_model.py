#!/usr/bin/env python3
"""
Quick test script to verify the classifier model loads correctly.
Run this locally before deploying to Hugging Face Space.
"""

import sys
import joblib
import sklearn

print("=" * 60)
print("Model Compatibility Test")
print("=" * 60)
print(f"Python version: {sys.version}")
print(f"scikit-learn version: {sklearn.__version__}")
print(f"joblib version: {joblib.__version__}")
print("=" * 60)

# Test loading the model
model_path = "finalfull_qari_classifier.pkl"

try:
    print(f"\nAttempting to load model from: {model_path}")
    clf = joblib.load(model_path)
    print(f"✓ Model loaded successfully!")
    print(f"✓ Model type: {type(clf).__name__}")
    
    # Try to get model info
    if hasattr(clf, 'classes_'):
        print(f"✓ Number of classes: {len(clf.classes_)}")
        print(f"✓ Classes: {clf.classes_[:5]}..." if len(clf.classes_) > 5 else f"✓ Classes: {clf.classes_}")
    
    if hasattr(clf, 'n_features_in_'):
        print(f"✓ Expected features: {clf.n_features_in_}")
    
    print("\n" + "=" * 60)
    print("✓ Model is compatible with current environment!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ Error loading model: {e}")
    print("\n" + "=" * 60)
    print("✗ Model is NOT compatible!")
    print("Please retrain the model with the current library versions.")
    print("=" * 60)
    sys.exit(1)
