import os
import numpy as np
import librosa
import torch
import joblib
import gradio as gr
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from speechbrain.pretrained import EncoderClassifier

# ----- Configuration -----
SR = 16000
MAX_SECONDS = 12
MAX_SAMPLES = SR * MAX_SECONDS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----- Load Models -----
print("Loading SpeechBrain ECAPA embedding model...")
embedder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": DEVICE}
).eval()

print("Loading Qari classifier...")
clf = joblib.load("finalfull_qari_classifier.pkl")
print("Model loaded successfully ✅")

# ----- Helper Functions -----
def load_audio_16k(path: str):
    y, _ = librosa.load(path, sr=SR, mono=True)
    if len(y) < MAX_SAMPLES:
        y = np.pad(y, (0, MAX_SAMPLES - len(y)))
    else:
        y = y[:MAX_SAMPLES]
    return y.astype(np.float32)

def extract_embedding(path: str):
    y = load_audio_16k(path)
    wav = torch.from_numpy(y).unsqueeze(0).unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        emb = embedder.encode_batch(wav).squeeze().detach().cpu().numpy()
    return emb.reshape(1, -1)

# ----- Gradio UI for manual testing -----
def predict_gradio(file):
    if file is None:
        return "No file provided."
    try:
        path = getattr(file, "name", file)
        X = extract_embedding(path)
        pred = str(clf.predict(X)[0])
        return f"Predicted Qari: {pred}"
    except Exception as e:
        return f"Error: {str(e)}"

ui = gr.Interface(
    fn=predict_gradio,
    inputs=gr.Audio(sources=["upload", "microphone"], type="filepath", label="Upload or record recitation"),
    outputs=gr.Textbox(label="Prediction"),
    title="Qari Recognizer",
    description="Upload or record a short recitation to identify the Qari."
)

# ----- FastAPI Endpoint for Expo App -----
api = FastAPI()

@api.post("/predict")
async def predict(audio: UploadFile = File(...)):
    try:
        temp_path = "temp_audio.m4a"
        with open(temp_path, "wb") as f:
            f.write(await audio.read())
        X = extract_embedding(temp_path)
        pred = str(clf.predict(X)[0])
        os.remove(temp_path)
        return JSONResponse({"qariName": pred})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Mount both (Gradio UI + API)
app = gr.mount_gradio_app(api, ui, path="/")
