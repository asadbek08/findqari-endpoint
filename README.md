# 🎵 Qari Recognizer

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green.svg)](https://fastapi.tiangolo.com)
[![Gradio](https://img.shields.io/badge/Gradio-4.44.0-orange.svg)](https://gradio.app)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A machine learning application that identifies Quranic reciters (Qaris) from audio recordings using SpeechBrain ECAPA embeddings and SVM classification.

## Features

- **Gradio UI**: Interactive web interface at `/` for uploading and testing audio files
- **FastAPI Endpoints**: 
  - `GET /health` - Health check endpoint
  - `POST /predict` - Audio prediction endpoint (accepts multipart form data)
- **CORS Support**: Configured for cross-origin requests
- **Docker Ready**: Containerized for easy deployment
- **Mobile App Compatible**: Perfect for React Native/Expo applications

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

## 🚀 Deployment Options

### Docker Deployment (Recommended)
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/qari-recognizer.git
cd qari-recognizer

# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t qari-recognizer .
docker run -p 7860:7860 qari-recognizer
```

### Cloud Platforms
The Docker container is ready for deployment on:
- **Google Cloud Run**: `gcloud run deploy --source .`
- **AWS ECS/Fargate**: Use the provided Dockerfile
- **Azure Container Instances**: Deploy from Docker Hub
- **Heroku**: Use container registry
- **Railway/Render**: Connect your GitHub repo

### Hugging Face Spaces
1. Create a new Space with `sdk: docker`
2. Push your code to the Space repository
3. HF will automatically build and deploy

## 📱 Mobile App Integration

Perfect for React Native/Expo apps:

```javascript
// Health check
const healthCheck = async () => {
  const response = await fetch('https://your-api-url.com/health');
  return response.json();
};

// Predict Qari from audio file
const predictQari = async (audioUri) => {
  const formData = new FormData();
  formData.append('audio', {
    uri: audioUri,
    type: 'audio/wav',
    name: 'recording.wav',
  });
  
  const response = await fetch('https://your-api-url.com/predict', {
    method: 'POST',
    body: formData,
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.json();
};
```

## 🧠 Model Details

- **Embedding Model**: SpeechBrain ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)
- **Classifier**: SVM trained on Qari embeddings
- **Audio Processing**: 16kHz mono, max 12 seconds
- **Device**: Automatically detects CUDA/CPU
- **Recognition Classes**: 50 renowned Quranic reciters (see [QARI_LABELS.txt](QARI_LABELS.txt))

### Supported Qaris (50 total)
The model can identify 50 different Quranic reciters including:
- **Imams of Masjid al-Haram**: Abdurrahman As-Sudais, Maher Al-Muaiqly, Saud Ash-Shuraim
- **Legendary Egyptian Reciters**: Abdul Basit Abdussomad, Al-Husary, Mohamed Siddiq El-Minshawi
- **Popular Contemporary**: Mishary Alafasy, Saad Al-Ghamdi, Abu Bakr Ash-Shatri
- **And 41 more renowned reciters** from Saudi Arabia, Egypt, Syria, UAE, Kuwait, and other countries

For the complete list with regional distribution, see [QARI_LABELS.txt](QARI_LABELS.txt).

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [SpeechBrain](https://speechbrain.github.io/) for the ECAPA-TDNN model
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Gradio](https://gradio.app/) for the UI components

## 📞 Support

If you have any questions or issues, please:
1. Check the [Issues](https://github.com/YOUR_USERNAME/qari-recognizer/issues) page
2. Create a new issue if needed
3. Provide detailed information about your problem
