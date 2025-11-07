#!/usr/bin/env python3
"""
Startup script for the Qari Recognizer application.
This script properly starts the FastAPI + Gradio app using uvicorn.
"""

import uvicorn
import os

if __name__ == "__main__":
    # Set environment variables for production
    os.environ["GRADIO_ALLOW_FLAGGING"] = "never"
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    
    print("Starting Qari Recognizer Server...")
    print("UI will be available at: http://localhost:7860/")
    print("API endpoints:")
    print("  - Health check: http://localhost:7860/health")
    print("  - Prediction: http://localhost:7860/predict")
    print("=" * 60)
    
    # Start the server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,  # Set to True for development
        log_level="info"
    )
