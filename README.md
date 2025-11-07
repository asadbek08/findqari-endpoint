---
title: Qari Recognizer
emoji: 🎵
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Qari Recognizer

A machine learning application that identifies Quranic reciters (Qaris) from audio recordings using SpeechBrain ECAPA embeddings and SVM classification.

## Features

- **Gradio UI**: Interactive web interface at `/` for uploading and testing audio files
- **FastAPI Endpoints**: 
  - `GET /health` - Health check endpoint
  - `POST /predict` - Audio prediction endpoint (accepts multipart form data)
- **CORS Support**: Configured for cross-origin requests
- **HF Spaces Compatible**: Works seamlessly on Hugging Face Spaces

## API Usage

### Health Check
```bash
curl http://localhost:7860/health
```

### Predict Qari
```bash
curl -X POST -F "audio=@your_audio.wav" http://localhost:7860/predict
```

Response format:
```json
{
  "qariName": "Abdul Basit Abdussomad",
  "device": "cpu"
}
```

## Supported Audio Formats
- WAV (.wav)
- MP3 (.mp3)
- M4A (.m4a)
- FLAC (.flac)
- OGG (.ogg)
- AAC (.aac)

## Local Development

### Option 1: Docker (Recommended)
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t qari-recognizer .
docker run -p 7860:7860 qari-recognizer
```

### Option 2: Python Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Run directly
python3 app.py

# Or with uvicorn
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Docker Deployment

### Hugging Face Spaces
1. Change SDK to `docker` in README header
2. Add `Dockerfile` to your Space
3. Push files - HF will automatically build and deploy

### Other Platforms
The Docker container is ready for deployment on:
- Google Cloud Run
- AWS ECS/Fargate
- Azure Container Instances
- Any Docker-compatible platform

## Model Details

- **Embedding Model**: SpeechBrain ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)
- **Classifier**: SVM trained on Qari embeddings
- **Audio Processing**: 16kHz mono, max 12 seconds
- **Device**: Automatically detects CUDA/CPU
