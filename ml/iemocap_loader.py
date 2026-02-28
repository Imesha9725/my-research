"""
Load IEMOCAP dataset for emotion (audio + labels).
Supports: (1) root/IEMOCAP/Session1..5/  (2) root/Session1..5/  (e.g. Kaggle extract)
Labels: neu, hap, ang, sad, exc, fru -> we map to 4 classes: neutral, happy, angry, sad (exc->happy, fru->angry).
"""
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional

# Standard IEMOCAP emotion set; we map to 4 classes for training
LABEL_MAP = {"neu": "neutral", "hap": "happy", "exc": "happy", "ang": "angry", "fru": "angry", "sad": "sad"}
VALID_LABELS = set(LABEL_MAP.keys())


def find_iemocap_root(data_path: str) -> Optional[Path]:
    """Return path to folder that contains Session1, Session2, ..."""
    root = Path(data_path).resolve()
    if not root.is_dir():
        return None
    # Case 1: root/IEMOCAP/Session1...
    iemocap = root / "IEMOCAP"
    if iemocap.is_dir():
        if (iemocap / "Session1").is_dir():
            return iemocap
    # Case 2: root/Session1...
    if (root / "Session1").is_dir():
        return root
    return None


def load_session_labels(session_dir: Path) -> dict:
    """Parse EmoEvaluation files in session_dir/dialog/EmoEvaluation/.
    Returns dict: wav_stem -> label (neu, hap, ang, sad, exc, fru).
    """
    label_dir = session_dir / "dialog" / "EmoEvaluation"
    if not label_dir.is_dir():
        return {}
    mapping = {}
    for txt_path in label_dir.glob("*.txt"):
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip().startswith("["):
                    continue
                parts = re.split(r"[\t\n]", line)
                if len(parts) < 3:
                    continue
                wav_stem = parts[1].strip()
                label = parts[2].strip().lower()
                if label in VALID_LABELS:
                    mapping[wav_stem] = label
    return mapping


def get_wav_paths(session_dir: Path) -> List[Tuple[str, Path]]:
    """Return list of (wav_stem, absolute_wav_path) under session_dir/sentences/wav/."""
    wav_dir = session_dir / "sentences" / "wav"
    if not wav_dir.is_dir():
        return []
    out = []
    for wav_path in wav_dir.rglob("*.wav"):
        stem = wav_path.stem
        out.append((stem, wav_path.resolve()))
    return out


def load_iemocap(data_path: str) -> List[Tuple[Path, str]]:
    """
    Load all (wav_path, emotion_label) from IEMOCAP.
    emotion_label is one of: neutral, happy, angry, sad.
    data_path: path to folder containing IEMOCAP or Session1..5.
    """
    root = find_iemocap_root(data_path)
    if root is None:
        return []

    samples = []
    for session_num in range(1, 6):
        session_name = f"Session{session_num}"
        session_dir = root / session_name
        if not session_dir.is_dir():
            continue
        labels = load_session_labels(session_dir)
        for stem, wav_path in get_wav_paths(session_dir):
            if stem not in labels:
                continue
            raw_label = labels[stem]
            mapped = LABEL_MAP.get(raw_label, "neutral")
            samples.append((wav_path, mapped))
    return samples


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("IEMOCAP_PATH", ".")
    samples = load_iemocap(path)
    print(f"Loaded {len(samples)} samples from {path}")
    if samples:
        from collections import Counter
        print("Label counts:", Counter(s[1] for s in samples))
