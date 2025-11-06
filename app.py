# app.py
# Qari Recognizer — Gradio UI + FastAPI JSON API
# Works on HF Spaces CPU (and GPU if available)
# Includes a compatibility shim for newer huggingface_hub versions.

import os
import io
import gc
import json
import numpy as np
import librosa
import torch
import joblib
import gradio as gr



os.environ["GRADIO_ALLOW_FLAGGING"] = "never"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_LAUNCH_METHOD"] = "spaces"  # 👈 important line


# ---- COMPAT SHIM (must be before importing speechbrain) -----------------------
# SpeechBrain<=0.5.16 still calls hf_hub_download(..., use_auth_token=...).
# Newer huggingface_hub removed that kwarg in favor of `token`.
try:
    import huggingface_hub as _hh  # type: ignore
    from huggingface_hub import hf_hub_download as _orig_hf_hub_download  # type: ignore

    def _compat_hf_hub_download(*args, **kwargs):
        if "use_auth_token" in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        return _orig_hf_hub_download(*args, **kwargs)

    _hh.hf_hub_download = _compat_hf_hub_download  # monkey-patch
except Exception as _e:
    # If this fails, requirements.txt pin should already fix it; continue anyway.
    print("INFO: hf_hub_download compat shim not applied:", _e)

# ------------------------------------------------------------------------------
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from speechbrain.pretrained import EncoderClassifier

# ===================== Config =====================
SR = 16000
MAX_SECONDS = 12
MAX_SAMPLES = SR * MAX_SECONDS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Where SpeechBrain will cache weights (speeds up cold starts)
SB_SAVEDIR = os.path.join("pretrained_models", "ecapa")

CLASSIFIER_PATH = "finalfull_qari_classifier.pkl"  # uploaded alongside this file

# ===================== Lazy loaders =====================
_embedder = None
_clf = None

def get_embedder():
    global _embedder
    if _embedder is None:
        print(f"Loading SpeechBrain ECAPA on {DEVICE} …")
        _embedder = EncoderClassifier.from_hparams(
            source="models/ecapa",
            run_opts={"device": DEVICE},
            savedir=SB_SAVEDIR,
        ).eval()
        print("ECAPA loaded ✅")
    return _embedder

def get_classifier():
    global _clf
    if _clf is None:
        print("Loading Qari SVM classifier …")
        _clf = joblib.load(CLASSIFIER_PATH)
        print("Classifier loaded ✅")
    return _clf

# ===================== DSP helpers =====================
def load_audio_16k(path: str) -> np.ndarray:
    """Load mono audio resampled to 16k and trim/pad to MAX_SECONDS."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    if len(y) < MAX_SAMPLES:
        y = np.pad(y, (0, MAX_SAMPLES - len(y)))
    elif len(y) > MAX_SAMPLES:
        y = y[:MAX_SAMPLES]
    return y.astype(np.float32)

def extract_embedding_from_path(path: str) -> np.ndarray:
    """ECAPA embedding for a file path -> shape (1, D)"""
    y = load_audio_16k(path)
    wav = torch.from_numpy(y).unsqueeze(0).unsqueeze(1).to(DEVICE)  # [1,1,T]
    emb = None
    with torch.no_grad():
        emb = get_embedder().encode_batch(wav).squeeze().detach().cpu().numpy()
    emb = emb.reshape(1, -1)
    # light cleanup
    del wav
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return emb

# ===================== Gradio UI (manual testing) =====================
def predict_gradio(file):
    if file is None:
        return "No file provided."
    try:
        path = getattr(file, "name", file)  # gradio may pass a tuple or path
        X = extract_embedding_from_path(path)
        pred = str(get_classifier().predict(X)[0])
        return f"Predicted Qari: {pred}"
    except Exception as e:
        return f"Error: {str(e)}"

ui = gr.Interface(
    fn=predict_gradio,
    inputs=gr.Audio(
        sources=["upload", "microphone"],
        type="filepath",
        label="Upload or record recitation (.wav/.m4a)"
    ),
    outputs=gr.Textbox(label="Prediction"),
    title="Qari Recognizer",
    description="Uploads a short recitation and predicts the Qari name."
)

# ===================== FastAPI JSON API =====================
api = FastAPI()

@api.get("/health")
async def health():
    return {"status": "ok", "device": DEVICE}

@api.post("/predict")
async def predict(audio: UploadFile = File(...)):
    """
    Multipart form-data:
      field: audio (file)
    Returns:
      { "qariName": "<string>" }
    """
    try:
        suffix = os.path.splitext(audio.filename or "")[1].lower()
        # Accept common audio types
        if suffix not in [".wav", ".m4a", ".mp3", ".flac", ".ogg", ".aac", ""]:
            return JSONResponse({"error": f"Unsupported file type: {suffix}"}, status_code=400)

        tmp_path = "tmp_audio" + (suffix if suffix else ".m4a")
        with open(tmp_path, "wb") as f:
            f.write(await audio.read())

        X = extract_embedding_from_path(tmp_path)
        pred = str(get_classifier().predict(X)[0])

        try:
            os.remove(tmp_path)
        except Exception:
            pass

        return JSONResponse({"qariName": pred, "device": DEVICE})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Mount Gradio UI at "/" and keep the JSON API on /predict
app = gr.mount_gradio_app(api, ui, path="/")

# Optional: preload on cold start to hide first-request latency
if os.environ.get("PRELOAD_ON_START", "1") == "1":
    try:
        _ = get_embedder()
        _ = get_classifier()
    except Exception as e:
        # If preload fails, Space will still boot; first request will trigger load.
        print("Preload warning:", e)
