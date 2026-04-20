#!/usr/bin/env python3
"""
Download and normalize public datasets for this project (text emotion + LoRA JSONL).

Uses Hugging Face `datasets` (no manual Kaggle unzip). Compatible with `datasets` 4.x
where legacy dataset *scripts* are disabled — we avoid `empathetic_dialogues` / `daily_dialog`
script repos and use mirrors / parquet-backed hubs instead.

Outputs (default: ml/data/processed/):
  - goemotions_for_text_emotion.csv     -> text,emotion (labels match train_text_emotion.py)
  - empathetic_dialogues_sft.jsonl      -> chat messages for finetune_lora_support.py
  - counsel_chat_sft.jsonl              -> same
  - merged_empathic_support_train.jsonl -> optional: hand-written JSONL + the two above

Examples:
  python ml/preprocess_external_datasets.py
  python ml/preprocess_external_datasets.py --max-per-source 2000
  python ml/preprocess_external_datasets.py --merge-base ml/data/empathic_support_train.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

# Same system line as ml/data/empathic_support_train.jsonl (keep in sync for LoRA)
DEFAULT_SYSTEM = (
    "You are a supportive, empathetic mental health companion. You listen without judgment. "
    "You are NOT a therapist or doctor—do not diagnose. Keep replies concise, warm, and natural. "
    "If someone may be in crisis, encourage professional and helpline support "
    "(e.g. Sumithrayo 1926 in Sri Lanka)."
)

# Must match ml/train_text_emotion.py LABELS
LABELS = ["neutral", "sad", "anxious", "stress", "angry", "lonely", "happy", "tired", "fear"]

# GoEmotions ClassLabel names (index order) -> our coarse labels for the text classifier
GOEMOTION_TO_LABEL: dict[str, str] = {
    "admiration": "happy",
    "amusement": "happy",
    "anger": "angry",
    "annoyance": "angry",
    "approval": "happy",
    "caring": "happy",
    "confusion": "anxious",
    "curiosity": "neutral",
    "desire": "neutral",
    "disappointment": "sad",
    "disapproval": "angry",
    "disgust": "angry",
    "embarrassment": "anxious",
    "excitement": "happy",
    "fear": "fear",
    "gratitude": "happy",
    "grief": "sad",
    "joy": "happy",
    "love": "happy",
    "nervousness": "anxious",
    "optimism": "happy",
    "pride": "happy",
    "realization": "neutral",
    "relief": "happy",
    "remorse": "sad",
    "sadness": "sad",
    "surprise": "neutral",
    "neutral": "neutral",
}


def clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    t = text.replace("\xa0", " ").replace("\u200b", "")
    t = re.sub(r"https?://\S+", " ", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def goemotion_indices_to_label(indices: list[int], names: list[str]) -> str | None:
    if not indices:
        return None
    mapped: list[str] = []
    for i in indices:
        if 0 <= i < len(names):
            src = names[i]
            tgt = GOEMOTION_TO_LABEL.get(src)
            if tgt:
                mapped.append(tgt)
    if not mapped:
        return None
    counts = Counter(mapped)
    return counts.most_common(1)[0][0]


def export_goemotions(out_csv: Path, max_rows: int | None) -> int:
    from datasets import load_dataset

    ds = load_dataset("go_emotions", "simplified", split="train")
    names = list(ds.features["labels"].feature.names)  # type: ignore[attr-defined]
    rows: list[tuple[str, str]] = []
    for i, ex in enumerate(ds):
        if max_rows is not None and len(rows) >= max_rows:
            break
        text = clean_text(ex.get("text") or "")
        if len(text) < 4:
            continue
        labs = ex.get("labels") or []
        if not isinstance(labs, list):
            continue
        lab = goemotion_indices_to_label(labs, names)
        if not lab:
            continue
        rows.append((text, lab))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "emotion"])
        w.writeheader()
        for t, e in rows:
            w.writerow({"text": t, "emotion": e})
    return len(rows)


def export_empathetic_jsonl(out_jsonl: Path, max_examples: int | None) -> int:
    """Turn-based empathetic data (parquet-backed mirror; see README)."""
    from datasets import load_dataset

    ds = load_dataset("pixelsandpointers/empathetic_dialogues_for_lm", split="train")
    n = 0
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for ex in ds:
            if max_examples is not None and n >= max_examples:
                break
            conv = ex.get("conv") or []
            if not isinstance(conv, list) or len(conv) < 2:
                continue
            turns = [clean_text(str(t).replace("_comma_", ",")) for t in conv]
            for i in range(len(turns) - 1):
                if max_examples is not None and n >= max_examples:
                    break
                u, a = turns[i], turns[i + 1]
                if len(u) < 8 or len(a) < 12:
                    continue
                obj = {
                    "messages": [
                        {"role": "system", "content": DEFAULT_SYSTEM},
                        {"role": "user", "content": u},
                        {"role": "assistant", "content": a},
                    ]
                }
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n += 1
    return n


def export_counsel_jsonl(out_jsonl: Path, max_examples: int | None) -> int:
    from datasets import load_dataset

    ds = load_dataset("nbertagnolli/counsel-chat", split="train")
    n = 0
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for ex in ds:
            if max_examples is not None and n >= max_examples:
                break
            title = clean_text(ex.get("questionTitle") or "")
            body = clean_text(ex.get("questionText") or "")
            ans = clean_text(ex.get("answerText") or "")
            if not ans or len(ans) < 40:
                continue
            user_parts = [p for p in (title, body) if p]
            user_msg = "\n\n".join(user_parts) if user_parts else body
            if len(user_msg) < 15:
                continue
            obj = {
                "messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM},
                    {"role": "user", "content": user_msg[:4000]},
                    {"role": "assistant", "content": ans[:8000]},
                ]
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n


def merge_jsonl_files(
    base_path: Path,
    extra_paths: list[Path],
    out_path: Path,
    seed: int,
) -> int:
    lines: list[str] = []
    with open(base_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    for p in extra_paths:
        if not p.is_file():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
    rng = random.Random(seed)
    rng.shuffle(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    return len(lines)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Preprocess GoEmotions, EmpatheticDialogues mirror, CounselChat")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "data" / "processed",
        help="Directory for CSV/JSONL outputs",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=None,
        help="Cap rows per source (for quick tests). None = use all available after filters.",
    )
    parser.add_argument(
        "--merge-base",
        type=Path,
        default=None,
        help="If set, write merged_empathic_support_train.jsonl = this file + ED + Counsel JSONL (shuffled).",
    )
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--skip-go", action="store_true")
    parser.add_argument("--skip-ed", action="store_true")
    parser.add_argument("--skip-counsel", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = args.max_per_source

    summary: dict[str, int] = {}

    if not args.skip_go:
        p = out_dir / "goemotions_for_text_emotion.csv"
        summary["goemotions_csv_rows"] = export_goemotions(p, cap)
        print(f"Wrote {summary['goemotions_csv_rows']} rows -> {p}")

    ed_path = out_dir / "empathetic_dialogues_sft.jsonl"
    if not args.skip_ed:
        summary["empathetic_jsonl_lines"] = export_empathetic_jsonl(ed_path, cap)
        print(f"Wrote {summary['empathetic_jsonl_lines']} lines -> {ed_path}")

    cc_path = out_dir / "counsel_chat_sft.jsonl"
    if not args.skip_counsel:
        summary["counsel_jsonl_lines"] = export_counsel_jsonl(cc_path, cap)
        print(f"Wrote {summary['counsel_jsonl_lines']} lines -> {cc_path}")

    if args.merge_base:
        if not args.merge_base.is_file():
            raise SystemExit(f"--merge-base not found: {args.merge_base}")
        merged = out_dir / "merged_empathic_support_train.jsonl"
        extra_files: list[Path] = []
        if not args.skip_ed and ed_path.is_file():
            extra_files.append(ed_path)
        if not args.skip_counsel and cc_path.is_file():
            extra_files.append(cc_path)
        total = merge_jsonl_files(args.merge_base, extra_files, merged, args.shuffle_seed)
        summary["merged_jsonl_lines"] = total
        print(f"Wrote {total} lines -> {merged} (base + {len(extra_files)} extra file(s), shuffled)")

    meta_path = out_dir / "preprocess_summary.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"labels_used": LABELS, "goemotion_map_keys": sorted(GOEMOTION_TO_LABEL.keys()), **summary}, f, indent=2)
    print(f"Summary -> {meta_path}")


if __name__ == "__main__":
    main()
