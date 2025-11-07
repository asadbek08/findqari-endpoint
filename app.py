import io
import sys
import os
import gc
import numpy as np
import librosa
import torch
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

# ===== Your inference code placeholder =====
ALLOWED_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}

def ext_from_name(name: str) -> str:
    i = name.rfind(".")
    return name[i:].lower() if i != -1 else ""

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

def load_audio_from_bytes(audio_bytes: bytes) -> np.ndarray:
    """Load audio from bytes using librosa."""
    try:
        # Use io.BytesIO to create a file-like object from bytes
        audio_buffer = io.BytesIO(audio_bytes)
        y, _ = librosa.load(audio_buffer, sr=SR, mono=True)
        print(f"Loaded audio shape: {y.shape}, dtype: {y.dtype}", file=sys.stderr)
        
        if len(y) < MAX_SAMPLES:
            # Use numpy padding instead of torch padding
            y = np.pad(y, (0, MAX_SAMPLES - len(y)), mode='constant', constant_values=0)
        elif len(y) > MAX_SAMPLES:
            y = y[:MAX_SAMPLES]
        
        print(f"Processed audio shape: {y.shape}", file=sys.stderr)
        return y.astype(np.float32)
    except Exception as e:
        print(f"Error loading audio from bytes: {e}", file=sys.stderr)
        raise

def extract_embedding_from_path(path: str) -> np.ndarray:
    y = load_audio_16k(path)
    print(f"Audio array shape before tensor: {y.shape}", file=sys.stderr)
    
    # Create tensor with proper dimensions for SpeechBrain ECAPA
    wav = torch.from_numpy(y).unsqueeze(0).to(DEVICE)  # [1, T] - batch dimension only
    print(f"Tensor shape: {wav.shape}, device: {wav.device}", file=sys.stderr)
    
    with torch.no_grad():
        try:
            emb = get_embedder().encode_batch(wav).squeeze().detach().cpu().numpy()
            print(f"Embedding shape: {emb.shape}", file=sys.stderr)
        except Exception as e:
            print(f"Error during embedding extraction: {e}", file=sys.stderr)
            # Try alternative tensor shape if first attempt fails
            wav_alt = wav.unsqueeze(1)  # [1, 1, T]
            print(f"Trying alternative tensor shape: {wav_alt.shape}", file=sys.stderr)
            emb = get_embedder().encode_batch(wav_alt).squeeze().detach().cpu().numpy()
            print(f"Alternative embedding shape: {emb.shape}", file=sys.stderr)
    
    emb = emb.reshape(1, -1)
    del wav
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return emb

def extract_embedding_from_bytes(audio_bytes: bytes) -> np.ndarray:
    """Extract embedding from audio bytes."""
    y = load_audio_from_bytes(audio_bytes)
    print(f"Audio array shape before tensor: {y.shape}", file=sys.stderr)
    
    # Create tensor with proper dimensions for SpeechBrain ECAPA
    wav = torch.from_numpy(y).unsqueeze(0).to(DEVICE)  # [1, T] - batch dimension only
    print(f"Tensor shape: {wav.shape}, device: {wav.device}", file=sys.stderr)
    
    with torch.no_grad():
        try:
            emb = get_embedder().encode_batch(wav).squeeze().detach().cpu().numpy()
            print(f"Embedding shape: {emb.shape}", file=sys.stderr)
        except Exception as e:
            print(f"Error during embedding extraction: {e}", file=sys.stderr)
            # Try alternative tensor shape if first attempt fails
            wav_alt = wav.unsqueeze(1)  # [1, 1, T]
            print(f"Trying alternative tensor shape: {wav_alt.shape}", file=sys.stderr)
            emb = get_embedder().encode_batch(wav_alt).squeeze().detach().cpu().numpy()
            print(f"Alternative embedding shape: {emb.shape}", file=sys.stderr)
    
    emb = emb.reshape(1, -1)
    del wav
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return emb

def recognize_qari_from_bytes(audio_bytes: bytes, filename: str) -> str:
    """Main inference function that takes audio bytes and returns Qari name."""
    try:
        X = extract_embedding_from_bytes(audio_bytes)
        print(f"Extracted embedding shape: {X.shape}", file=sys.stderr)
        
        pred = str(get_classifier().predict(X)[0])
        print(f"Prediction: {pred}", file=sys.stderr)
        
        return pred
    except Exception as e:
        print(f"Recognition error: {str(e)}", file=sys.stderr)
        raise

# ===== Gradio UI (optional) =====
def gradio_predict(file_path):
    # Gradio's Audio(type="filepath") gives a path; read bytes and reuse same model function
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    qari_name = recognize_qari_from_bytes(audio_bytes, file_path)
    return qari_name

demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Audio(type="filepath", label="Upload or record audio"),
    outputs=gr.Textbox(label="Predicted Qari"),
    title="Qari Recognizer",
)

# ===== FastAPI app with CORS =====
app = FastAPI(title="Qari Recognizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}

@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    ext = ext_from_name(audio.filename or "")
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext or '(no extension)'}")
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    qari_name = recognize_qari_from_bytes(audio_bytes, audio.filename or "audio")
    return {"qariName": qari_name, "device": DEVICE}

# Mount UI at root, keep API routes as-is
app = gr.mount_gradio_app(app, demo, path="/")

print("=" * 60, file=sys.stderr)
print("✓ Application initialized successfully!", file=sys.stderr)
print(f"✓ Device: {DEVICE}", file=sys.stderr)
print(f"✓ Models will load on first request (lazy loading)", file=sys.stderr)
print("✓ UI available at /", file=sys.stderr)
print("✓ API endpoints: /health and /predict", file=sys.stderr)
print("=" * 60, file=sys.stderr)

# For Docker deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"Starting server on {host}:{port}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
