"""
Emotion prediction API using IEMOCAP-trained SER model.
POST /predict_emotion with multipart audio file (WAV) or JSON { "text": "..." } for text-only.
Returns { "emotion": "neutral|happy|angry|sad", "source": "voice|text", "confidence": 0.9 }
Run: uvicorn ml.app:app --app-dir . --host 0.0.0.0 --port 5002
Or from repo root: python -m uvicorn ml.app:app --reload --port 5002
"""
import os
import io
import json
import tempfile
from pathlib import Path
import numpy as np
import librosa
import joblib
import base64
from fastapi import FastAPI, File, UploadFile, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel

app = FastAPI(title="Emotion API (IEMOCAP)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_DIR = Path(__file__).resolve().parent / "models"
scaler = None
model = None
label_encoder = None
config = {}
_text_tokenizer = None
_text_classifier = None

# Text emotion keywords; order matters: check sad (and "not good" etc.) before happy
TEXT_KEYWORDS_ORDERED = [
    ("sad", ["sad", "sadness", "depressed", "down", "miserable", "hopeless", "lonely", "not good", "not well", "not okay", "not great", "not fine", "no good", "don't feel good", "don't feel well", "i'm not good", "am not good", "not doing good", "not doing well"]),
    ("angry", ["angry", "anger", "mad", "frustrated", "annoyed"]),
    ("happy", ["happy", "good", "great", "better", "relieved", "hopeful", "excited", "joy"]),
]
SR = 16000
N_MFCC = 40
MAX_LEN = 100


def load_models():
    global scaler, model, label_encoder, config
    if (MODEL_DIR / "ser_model.joblib").exists():
        scaler = joblib.load(MODEL_DIR / "ser_scaler.joblib")
        model = joblib.load(MODEL_DIR / "ser_model.joblib")
        label_encoder = joblib.load(MODEL_DIR / "ser_label_encoder.joblib")
        with open(MODEL_DIR / "ser_config.json") as f:
            config = json.load(f)
        return True
    return False


def load_text_emotion_model():
    """Optional DistilBERT/BERT classifier from ml/models/text_emotion (train_text_emotion.py)."""
    global _text_tokenizer, _text_classifier
    path = MODEL_DIR / "text_emotion"
    if not (path / "config.json").exists():
        return False
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        print("Text emotion model found but torch/transformers not installed. pip install -r ml/requirements-text-emotion.txt")
        return False
    try:
        _text_tokenizer = AutoTokenizer.from_pretrained(str(path))
        _text_classifier = AutoModelForSequenceClassification.from_pretrained(str(path))
        _text_classifier.eval()
        print("Text emotion model loaded from", path)
        return True
    except Exception as e:
        print("Failed to load text emotion model:", e)
        _text_tokenizer = None
        _text_classifier = None
        return False


def predict_text(text: str):
    raw = (text or "").strip()
    if not raw:
        return "neutral", 0.5
    if _text_classifier is not None and _text_tokenizer is not None:
        try:
            import torch

            inputs = _text_tokenizer(raw, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                logits = _text_classifier(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1)
            idx = int(torch.argmax(probs).item())
            lid = _text_classifier.config.id2label
            if isinstance(lid, dict):
                label = lid.get(idx, lid.get(str(idx), "neutral"))
            else:
                label = lid[idx] if idx < len(lid) else "neutral"
            conf = float(probs[idx].item())
            return str(label).lower(), conf
        except Exception:
            pass
    t = raw.lower()
    for emotion, keywords in TEXT_KEYWORDS_ORDERED:
        if any(k in t for k in keywords):
            return emotion, 0.85
    return "neutral", 0.6


def extract_mfcc(wav_path_or_bytes, sr=SR, n_mfcc=N_MFCC, max_frames=MAX_LEN):
    try:
        if isinstance(wav_path_or_bytes, (str, Path)):
            y, _ = librosa.load(str(wav_path_or_bytes), sr=sr, mono=True)
        else:
            y, _ = librosa.load(io.BytesIO(wav_path_or_bytes), sr=sr, mono=True)
    except Exception:
        return None
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=512, hop_length=256)
    if mfcc.shape[1] >= max_frames:
        mfcc = mfcc[:, :max_frames]
    else:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_frames - mfcc.shape[1])), mode="constant", constant_values=0)
    return mfcc.flatten()


def predict_voice(audio_bytes: bytes):
    if model is None:
        return "neutral", 0.5
    x = extract_mfcc(audio_bytes)
    if x is None:
        return "neutral", 0.5  # e.g. unsupported format (webm may need ffmpeg)
    x = scaler.transform(x.reshape(1, -1))
    proba = model.predict_proba(x)[0]
    idx = np.argmax(proba)
    label = label_encoder.inverse_transform([idx])[0]
    return label, float(proba[idx])


@app.on_event("startup")
def startup():
    load_models()
    if model is not None:
        print("SER model loaded from", MODEL_DIR)
    else:
        print("No SER model found. Train with: IEMOCAP_PATH=/path/to/iemocap python ml/train_ser.py")
    load_text_emotion_model()
    if _text_classifier is None:
        print("Text emotion: using keyword fallback. Train with: python ml/train_text_emotion.py")


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "ser_loaded": model is not None,
        "text_emotion_model_loaded": _text_classifier is not None,
    }


class PredictBody(BaseModel):
    text: Optional[str] = None
    audio_base64: Optional[str] = None


@app.post("/api/predict_emotion")
async def predict_emotion(
    audio: UploadFile = File(None),
    text: str = Form(None),
):
    """Multipart: audio file + optional text."""
    results = []
    if audio and audio.filename:
        content = await audio.read()
        if len(content) > 0:
            em, conf = predict_voice(content)
            results.append({"emotion": em, "source": "voice", "confidence": conf})
    if text and text.strip():
        em, conf = predict_text(text.strip())
        results.append({"emotion": em, "source": "text", "confidence": conf})
    if not results:
        return {"emotions": [{"emotion": "neutral", "source": "none", "confidence": 0.5}], "primary": "neutral"}
    primary = next((r["emotion"] for r in results if r["source"] == "voice"), results[0]["emotion"])
    return {"emotions": results, "primary": primary}


@app.post("/api/predict_emotion_json")
async def predict_emotion_json(body: PredictBody = Body(...)):
    """JSON: text and/or audio_base64 (for Node backend)."""
    results = []
    if body.audio_base64:
        try:
            raw = base64.b64decode(body.audio_base64, validate=True)
            if len(raw) > 0:
                em, conf = predict_voice(raw)
                results.append({"emotion": em, "source": "voice", "confidence": conf})
        except Exception:
            pass
    if body.text and body.text.strip():
        em, conf = predict_text(body.text.strip())
        results.append({"emotion": em, "source": "text", "confidence": conf})
    if not results:
        return {"emotions": [{"emotion": "neutral", "source": "none", "confidence": 0.5}], "primary": "neutral"}
    primary = next((r["emotion"] for r in results if r["source"] == "voice"), results[0]["emotion"])
    return {"emotions": results, "primary": primary}
