"""
Train a Speech Emotion Recognition (SER) model on IEMOCAP.
Extracts MFCCs, trains a classifier, saves model + scaler to ml/models/.
Run from project root: python ml/train_ser.py
Set IEMOCAP_PATH to your dataset root (e.g. folder containing IEMOCAP or Session1..5).
"""
import os
import sys
import json
from pathlib import Path
import numpy as np
import librosa
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from iemocap_loader import load_iemocap, find_iemocap_root

# Feature config
SR = 16000
N_MFCC = 40
MAX_LEN = 100  # frames per utterance (pad/trim)
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def extract_mfcc(wav_path: Path, sr: int = SR, n_mfcc: int = N_MFCC, max_frames: int = MAX_LEN) -> np.ndarray:
    """Load WAV, compute MFCCs, return fixed-size vector (flatten and pad/trim)."""
    y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=512, hop_length=256)
    # (n_mfcc, time) -> flatten or take stats
    if mfcc.shape[1] >= max_frames:
        mfcc = mfcc[:, :max_frames]
    else:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_frames - mfcc.shape[1])), mode="constant", constant_values=0)
    return mfcc.flatten()


def main():
    data_path = os.environ.get("IEMOCAP_PATH", ".")
    root = find_iemocap_root(data_path)
    if root is None:
        print("IEMOCAP not found. Set IEMOCAP_PATH to the folder containing IEMOCAP or Session1..5.")
        print("Example: IEMOCAP_PATH=/path/to/your/iemocap-emotion-speech-database python ml/train_ser.py")
        sys.exit(1)

    print(f"Loading IEMOCAP from {root}...")
    samples = load_iemocap(str(root))
    if not samples:
        print("No samples found. Check path and folder structure (Session1/sentences/wav/, Session1/dialog/EmoEvaluation/).")
        sys.exit(1)

    print(f"Extracting MFCCs for {len(samples)} samples...")
    X = np.array([extract_mfcc(p) for p, _ in samples])
    y = np.array([label for _, label in samples])

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)
    print("Classes:", le.classes_.tolist())

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("Training MLP...")
    clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=200, random_state=42)
    clf.fit(X_train_s, y_train)
    acc = clf.score(X_test_s, y_test)
    print(f"Test accuracy: {acc:.3f}")

    joblib.dump(scaler, MODEL_DIR / "ser_scaler.joblib")
    joblib.dump(clf, MODEL_DIR / "ser_model.joblib")
    joblib.dump(le, MODEL_DIR / "ser_label_encoder.joblib")
    with open(MODEL_DIR / "ser_config.json", "w") as f:
        json.dump({"sr": SR, "n_mfcc": N_MFCC, "max_frames": MAX_LEN, "classes": le.classes_.tolist()}, f, indent=2)
    print(f"Saved model and config to {MODEL_DIR}")


if __name__ == "__main__":
    main()
