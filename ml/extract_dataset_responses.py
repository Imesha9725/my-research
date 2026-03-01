"""
Extract emotion -> response pairs from IEMOCAP_full_release.
Uses: Session*/dialog/EmoEvaluation/*.txt (emotion labels per utterance)
      Session*/dialog/transcriptions/*.txt (utterance text, in correct dialogue order)
Transcription files contain the FULL dialogue with both speakers alternating (A, B, A, B).
EmoEvaluation files are split by speaker—we use transcription order for correct user->response pairing.
Saves to ml/models/dataset_responses.json for use by the chatbot.

Run: IEMOCAP_PATH="D:\\My Research Project\\IEMOCAP_full_release" python ml/extract_dataset_responses.py
"""
import os
import re
import json
from pathlib import Path
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from iemocap_loader import find_iemocap_root, LABEL_MAP

# Accept all IEMOCAP labels; map extras to neutral
EXTRA_LABELS = {"sur": "neutral", "xxx": "neutral", "dis": "sad"}
FULL_LABEL_MAP = {**LABEL_MAP, **EXTRA_LABELS}

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = MODEL_DIR / "dataset_responses.json"


def load_utt_id_to_emotion(root: Path) -> dict:
    """
    Build utt_id -> emotion lookup from EmoEvaluation files.
    Format: [start - end]  utt_id  emotion  [V, A, D]
    """
    utt_emotion = {}
    for session_num in range(1, 6):
        session_dir = root / f"Session{session_num}"
        if not session_dir.is_dir():
            continue
        label_dir = session_dir / "dialog" / "EmoEvaluation"
        if not label_dir.is_dir():
            continue
        for txt_path in label_dir.glob("*.txt"):
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line.startswith("["):
                        continue
                    match = re.match(r"\[\s*[\d.]+\s*-\s*[\d.]+\s*\]\s+(\S+)\s+(\w+)", line)
                    if match:
                        utt_id, label = match.group(1), match.group(2).lower()
                        utt_emotion[utt_id] = FULL_LABEL_MAP.get(label, "neutral")
    return utt_emotion


def load_dialogues_in_order(root: Path) -> list:
    """
    Load each dialogue's utterances in correct turn order from transcription files.
    Transcription files have both speakers interleaved (A, B, A, B...).
    Returns list of [(utt_id, text), ...] per dialogue, one list per unique dialogue.
    """
    line_re = re.compile(r"^(\S+)\s+\[[^\]]+\]:\s*(.+)$")
    processed = set()
    all_dialogues = []
    for session_num in range(1, 6):
        session_dir = root / f"Session{session_num}"
        if not session_dir.is_dir():
            continue
        trans_dir = session_dir / "dialog" / "transcriptions"
        if not trans_dir.is_dir():
            continue
        for txt_path in sorted(trans_dir.glob("*.txt")):
            stem = txt_path.stem
            dialogue_id = (session_num, re.sub(r"^Ses\d{2}[FM]_", "", stem))
            if dialogue_id in processed:
                continue
            processed.add(dialogue_id)
            utterances = []
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    match = line_re.match(line)
                    if match:
                        utt_id, text = match.group(1), match.group(2).strip()
                        if text and len(text) > 2:
                            utterances.append((utt_id, text))
            if utterances:
                all_dialogues.append(utterances)
    return all_dialogues


def extract_responses(data_path: str) -> dict:
    """Build (user_utterance, emotion, response) pairs for input-output matching."""
    root = find_iemocap_root(data_path)
    if root is None:
        print("IEMOCAP not found. Set IEMOCAP_PATH to your dataset folder.")
        return {}
    utt_emotion = load_utt_id_to_emotion(root)
    dialogues = load_dialogues_in_order(root)
    if not dialogues:
        print("No transcriptions found in dialog/transcriptions/")
        return {}
    print(f"Found {len(utt_emotion)} emotion labels, {len(dialogues)} dialogues")
    pairs_by_emotion = defaultdict(list)
    for utterances in dialogues:
        for i in range(len(utterances) - 1):
            utt_id, user_text = utterances[i]
            next_id, reply = utterances[i + 1]
            emotion = utt_emotion.get(utt_id, "neutral")
            if user_text and reply and len(user_text) > 2 and len(reply) > 2:
                pairs_by_emotion[emotion].append({"user": user_text, "response": reply})
    return {k: v for k, v in pairs_by_emotion.items()}


def main():
    data_path = os.environ.get("IEMOCAP_PATH", ".")
    root = find_iemocap_root(data_path)
    if root is None:
        print("IEMOCAP not found. Set IEMOCAP_PATH to your dataset folder.")
        print('Example: $env:IEMOCAP_PATH="D:\\My Research Project\\IEMOCAP_full_release"')
        sys.exit(1)
    print(f"Extracting from {root}...")
    responses = extract_responses(str(root))
    if not responses:
        print("No emotion->response pairs extracted. Check folder structure.")
        sys.exit(1)
    for emo, lst in responses.items():
        print(f"  {emo}: {len(lst)} (user, response) pairs")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
