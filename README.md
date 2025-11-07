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

- **Gradio UI**: Interactive web interface for uploading and testing audio files
- **FastAPI Endpoints**: 
  - `GET /health` - Health check endpoint
  - `POST /predict` - Audio prediction endpoint (accepts multipart form data)
- **CORS Support**: Configured for cross-origin requests
- **Mobile App Compatible**: Perfect for React Native/Expo applications

## API Usage

### Health Check
```bash
curl -H "Authorization: Bearer YOUR_HF_TOKEN" https://asadbekiskandarov-qari-recognizer.hf.space/health
```

### Predict Qari
```bash
curl -X POST -H "Authorization: Bearer YOUR_HF_TOKEN" -F "audio=@your_audio.wav" https://asadbekiskandarov-qari-recognizer.hf.space/predict
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

## Model Details

- **Embedding Model**: SpeechBrain ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)
- **Classifier**: SVM trained on Qari embeddings
- **Audio Processing**: 16kHz mono, max 12 seconds
- **Device**: Automatically detects CUDA/CPU

## Recent Updates

- ✅ Fixed audio format recognition issues
- ✅ Added robust temporary file fallback for audio loading
- ✅ Fixed Gradio interface type handling
- ✅ Enhanced error handling and logging
