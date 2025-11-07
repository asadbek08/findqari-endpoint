---
title: Qari Recognizer
emoji: 🎵
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
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

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python3 start_server.py
```

Or use uvicorn directly:
```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Model Details

- **Embedding Model**: SpeechBrain ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)
- **Classifier**: SVM trained on Qari embeddings
- **Audio Processing**: 16kHz mono, max 12 seconds
- **Device**: Automatically detects CUDA/CPU
