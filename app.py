# app.py — minimal, non-blocking startup (HF Spaces friendly)

import sys
import os
import gc
import numpy as np
import librosa
import torch
import joblib
import gradio as gr

print("=" * 60, file=sys.stderr)
print("Starting Qari Recognizer App...", file=sys.stderr)
print(f"Python: {sys.version}", file=sys.stderr)
print(f"PyTorch: {torch.__version__}", file=sys.stderr)
print(f"NumPy: {np.__version__}", file=sys.stderr)
print("=" * 60, file=sys.stderr)

# --- Gradio/Spaces runtime flags (avoid double-launch & analytics)
os.environ["GRADIO_ALLOW_FLAGGING"] = "never"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_LAUNCH_METHOD"] = "spaces"

# ---- COMPAT SHIM (before importing speechbrain)
# SpeechBrain<=0.5.16 used use_auth_token; newer hub renamed to token.
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

try:
    from fastapi import FastAPI, UploadFile, File
    from fastapi.responses import JSONResponse
    print("✓ FastAPI imported", file=sys.stderr)
except Exception as e:
    print(f"✗ FastAPI import error: {e}", file=sys.stderr)
    raise

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    print("✓ Hugging Face Hub imported", file=sys.stderr)
except Exception as e:
    print(f"✗ HF Hub import error: {e}", file=sys.stderr)
    raise

try:
    from speechbrain.pretrained import EncoderClassifier
    print("✓ SpeechBrain imported", file=sys.stderr)
except Exception as e:
    print(f"✗ SpeechBrain import error: {e}", file=sys.stderr)
    raise

# ===================== Config =====================
SR = 16000
MAX_SECONDS = 12
MAX_SAMPLES = SR * MAX_SECONDS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSIFIER_PATH = "finalfull_qari_classifier.pkl"   # uploaded alongside app.py
ECAPA_LOCAL_DIR = "models/ecapa"                    # local folder for ECAPA
SB_SAVEDIR = os.path.join("pretrained_models", "ecapa_cache")

NEEDED_FILES = [
    "hyperparams.yaml",
    "embedding_model.ckpt",
    "classifier.ckpt",
    "mean_var_norm_emb.ckpt",
    "label_encoder.txt",
]

# ===================== Lazy, on-demand loaders =====================
_embedder = None
_clf = None

def ensure_ecapa_local():
    """Ensure SpeechBrain ECAPA files exist locally (download once if missing)."""
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

def get_embedder():
    """Load SpeechBrain ECAPA only when first needed."""
    global _embedder
    if _embedder is None:
        ensure_ecapa_local()
        print(f"Loading SpeechBrain ECAPA from {ECAPA_LOCAL_DIR} on {DEVICE} …")
        _embedder = EncoderClassifier.from_hparams(
            source=ECAPA_LOCAL_DIR,     # local path (no hub fetch during encode)
            run_opts={"device": DEVICE},
            savedir=SB_SAVEDIR,
        ).eval()
        print("ECAPA loaded ✅")
    return _embedder

def get_classifier():
    """Load your SVM only when first needed."""
    global _clf
    if _clf is None:
        print("Loading Qari SVM classifier …", file=sys.stderr)
        
        classifier_file = CLASSIFIER_PATH
        
        # Check if local file exists and if it's a Git LFS pointer
        if os.path.exists(CLASSIFIER_PATH):
            with open(CLASSIFIER_PATH, 'rb') as f:
                first_line = f.readline().decode('utf-8', errors='ignore').strip()
                if first_line.startswith('version https://git-lfs.github.com'):
                    print("Local file is Git LFS pointer, downloading from Hub...", file=sys.stderr)
                    try:
                        # Try to get the space name from environment or use a fallback
                        space_name = os.environ.get('SPACE_ID', 'asadbekiskandarov/qari-recognizer')
                        classifier_file = hf_hub_download(
                            repo_id=space_name,
                            filename="finalfull_qari_classifier.pkl",
                            repo_type="space"
                        )
                        print(f"Downloaded classifier from Hub to: {classifier_file}", file=sys.stderr)
                    except Exception as e:
                        print(f"Failed to download from Hub: {e}", file=sys.stderr)
                        raise RuntimeError(f"Classifier file is Git LFS pointer and download failed: {e}")
        else:
            raise FileNotFoundError(f"Classifier file not found: {CLASSIFIER_PATH}")
        
        try:
            _clf = joblib.load(classifier_file)
            print(f"Classifier loaded ✅ (type: {type(_clf).__name__})", file=sys.stderr)
        except Exception as e:
            print(f"Error loading classifier: {e}", file=sys.stderr)
            print(f"This may be due to version incompatibility.", file=sys.stderr)
            print(f"Consider retraining the model with current library versions.", file=sys.stderr)
            raise RuntimeError(f"Failed to load classifier: {e}. Model may be incompatible with current scikit-learn version.")
    return _clf

# ===================== DSP helpers =====================
def load_audio_16k(path: str) -> np.ndarray:
    try:
        y, _ = librosa.load(path, sr=SR, mono=True)
        print(f"Loaded audio shape: {y.shape}, dtype: {y.dtype}", file=sys.stderr)
        
        if len(y) < MAX_SAMPLES:
            # Use numpy padding instead of torch padding
            y = np.pad(y, (0, MAX_SAMPLES - len(y)), mode='constant', constant_values=0)
        elif len(y) > MAX_SAMPLES:
            y = y[:MAX_SAMPLES]
        
        print(f"Processed audio shape: {y.shape}", file=sys.stderr)
        return y.astype(np.float32)
    except Exception as e:
        print(f"Error loading audio: {e}", file=sys.stderr)
        raise

def extract_embedding_from_path(path: str) -> np.ndarray:
    y = load_audio_16k(path)
    wav = torch.from_numpy(y).unsqueeze(0).unsqueeze(1).to(DEVICE)  # [1,1,T]
    with torch.no_grad():
        emb = get_embedder().encode_batch(wav).squeeze().detach().cpu().numpy()
    emb = emb.reshape(1, -1)
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
        path = getattr(file, "name", file)  # gradio may pass a path/tempfile
        print(f"Processing file: {path}", file=sys.stderr)
        
        X = extract_embedding_from_path(path)
        print(f"Extracted embedding shape: {X.shape}", file=sys.stderr)
        
        pred = str(get_classifier().predict(X)[0])
        print(f"Prediction: {pred}", file=sys.stderr)
        
        return f"Predicted Qari: {pred}"
    except Exception as e:
        print(f"Prediction error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

ui = gr.Interface(
    fn=predict_gradio,
    inputs=gr.Audio(
        sources=["upload", "microphone"],
        type="filepath",
        label="Upload or record recitation (.wav/.m4a/.mp3/.flac/.ogg/.aac)"
    ),
    outputs=gr.Textbox(label="Prediction"),
    title="Qari Recognizer",
    description="Uploads a short recitation and predicts the Qari name."
)

# Export a Gradio variable so the SDK detects a Gradio app
demo = ui

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

# ===================== Mount app (NO preload at import) =====================
# Export ASGI app for Spaces (FastAPI with Gradio mounted at "/")
print("Mounting Gradio app to FastAPI...", file=sys.stderr)
app = gr.mount_gradio_app(api, demo, path="/")

print("=" * 60, file=sys.stderr)
print("✓ Application initialized successfully!", file=sys.stderr)
print(f"✓ Device: {DEVICE}", file=sys.stderr)
print(f"✓ Models will load on first request (lazy loading)", file=sys.stderr)
print("=" * 60, file=sys.stderr)

# No warmup here. Models load on first request to avoid blocking initialization.
