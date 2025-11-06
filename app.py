# app.py
# Qari Recognizer — Gradio UI + FastAPI JSON API (HF Spaces friendly)

import os
import gc
import numpy as np
import librosa
import torch
import joblib
import gradio as gr

# --- Gradio/Spaces runtime flags (avoid double-launch, keep things lean)
os.environ["GRADIO_ALLOW_FLAGGING"] = "never"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_LAUNCH_METHOD"] = "spaces"

# ---- COMPAT SHIM (before importing speechbrain)
# SpeechBrain<=0.5.16 calls hf_hub_download(..., use_auth_token=...).
# Newer huggingface_hub removed that kwarg; map to `token` if present.
try:
    import huggingface_hub as _hh  # type: ignore
    from huggingface_hub import hf_hub_download as _orig_hf_hub_download  # type: ignore

    def _compat_hf_hub_download(*args, **kwargs):
        if "use_auth_token" in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        return _orig_hf_hub_download(*args, **kwargs)

    _hh.hf_hub_download = _compat_hf_hub_download
except Exception as _e:
    print("INFO: hf_hub_download compat shim not applied:", _e)

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from huggingface_hub import snapshot_download
from speechbrain.pretrained import EncoderClassifier

# ===================== Config =====================
SR = 16000
MAX_SECONDS = 12
MAX_SAMPLES = SR * MAX_SECONDS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSIFIER_PATH = "finalfull_qari_classifier.pkl"  # upload this alongside app.py
ECAPA_LOCAL_DIR = "models/ecapa"                   # local cache folder for ECAPA
SB_SAVEDIR = os.path.join("pretrained_models", "ecapa_cache")  # extra cache

# ===================== Ensure ECAPA files exist locally =====================
NEEDED_FILES = [
    "hyperparams.yaml",
    "embedding_model.ckpt",
    "classifier.ckpt",
    "mean_var_norm_emb.ckpt",
    "label_encoder.txt",
]

def ensure_ecapa_local():
    """Download SpeechBrain ECAPA model files locally if missing."""
    ok = os.path.isdir(ECAPA_LOCAL_DIR) and all(
        os.path.isfile(os.path.join(ECAPA_LOCAL_DIR, f)) for f in NEEDED_FILES
    )
    if not ok:
        os.makedirs(ECAPA_LOCAL_DIR, exist_ok=True)
        print("Downloading ECAPA files locally …")
        snapshot_download(
            repo_id="speechbrain/spkrec-ecapa-voxceleb",
            local_dir=ECAPA_LOCAL_DIR,
            local_dir_use_symlinks=False,
            allow_patterns=["hyperparams.yaml", "*.ckpt", "label_encoder.txt"],
        )
        print("ECAPA files ready at:", ECAPA_LOCAL_DIR)

# ===================== Lazy loaders =====================
_embedder = None
_clf = None

def get_embedder():
    """Load SpeechBrain ECAPA (once)."""
    global _embedder
    if _embedder is None:
        ensure_ecapa_local()
        print(f"Loading SpeechBrain ECAPA from {ECAPA_LOCAL_DIR} on {DEVICE} …")
        _embedder = EncoderClassifier.from_hparams(
            source=ECAPA_LOCAL_DIR,          # local path (no Hub fetch during encode)
            run_opts={"device": DEVICE},
            savedir=SB_SAVEDIR,
        ).eval()
        print("ECAPA loaded ✅")
    return _embedder

def get_classifier():
    """Load your trained SVM (once)."""
    global _clf
    if _clf is None:
        print("Loading Qari SVM classifier …")
        _clf = joblib.load(CLASSIFIER_PATH)
        print("Classifier loaded ✅")
    return _clf

# ===================== DSP helpers =====================
def load_audio_16k(path: str) -> np.ndarray:
    """Load mono audio at 16kHz and trim/pad to MAX_SECONDS."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    if len(y) < MAX_SAMPLES:
        y = np.pad(y, (0, MAX_SAMPLES - len(y)))
    elif len(y) > MAX_SAMPLES:
        y = y[:MAX_SAMPLES]
    return y.astype(np.float32)

def extract_embedding_from_path(path: str) -> np.ndarray:
    """Return ECAPA embedding (1, D) for a given file path."""
    y = load_audio_16k(path)
    wav = torch.from_numpy(y).unsqueeze(0).unsqueeze(1).to(DEVICE)  # [1,1,T]
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
        path = getattr(file, "name", file)  # gradio may pass a path or tempfile
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
        label="Upload or record recitation (.wav/.m4a/.mp3)"
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
    Returns JSON:
      { "qariName": "<string>", "device": "cpu|cuda" }
    """
    try:
        suffix = os.path.splitext(audio.filename or "")[1].lower()
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

# ===================== Mount app & non-blocking warmup =====================
# Mount Gradio UI at "/" and keep JSON API on /predict
app = gr.mount_gradio_app(api, ui, path="/")

# Warm up in background so Spaces initializes immediately
import threading
def _warmup():
    try:
        _ = get_embedder()
        _ = get_classifier()
        print("Warmup finished ✅")
    except Exception as e:
        print("Warmup failed (will load on first request):", e)

if os.environ.get("PRELOAD_ON_START", "1") == "1":
    threading.Thread(target=_warmup, daemon=True).start()
