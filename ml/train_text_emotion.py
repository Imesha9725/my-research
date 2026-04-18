#!/usr/bin/env python3
"""
Train a text emotion classifier (DistilBERT / BERT) for mental-health chat.

Data sources:
  1) IEMOCAP-derived user lines from ml/models/dataset_responses.json (labels: happy, sad, angry, neutral)
  2) Optional augment CSV (default ml/data/text_emotion_augment.csv) for anxious, stress, lonely, tired, fear

Aligns labels with server KEYWORD_MAP-style emotions. Speech stays on SER; this improves TEXT emotion.

Install:
  pip install -r ml/requirements.txt -r ml/requirements-text-emotion.txt

Run:
  python ml/train_text_emotion.py
  python ml/train_text_emotion.py --max-per-class 800 --epochs 2
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
DATASET_JSON = MODELS / "dataset_responses.json"
DEFAULT_AUGMENT = ROOT / "data" / "text_emotion_augment.csv"
OUTPUT_DIR = MODELS / "text_emotion"

# Must match server/ml keyword emotions + neutral (order = label id)
LABELS = ["neutral", "sad", "anxious", "stress", "angry", "lonely", "happy", "tired", "fear"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}


def load_rows_from_dataset_responses(path: Path, max_per_class: int | None) -> list[tuple[str, str]]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows: list[tuple[str, str]] = []
    for emo, pairs in data.items():
        if emo not in LABEL2ID:
            continue
        if not isinstance(pairs, list):
            continue
        got: list[tuple[str, str]] = []
        for p in pairs:
            if not isinstance(p, dict):
                continue
            u = (p.get("user") or "").strip()
            if len(u) < 3:
                continue
            got.append((u, emo))
        random.shuffle(got)
        if max_per_class is not None:
            got = got[:max_per_class]
        rows.extend(got)
    return rows


def load_augment_csv(path: Path) -> list[tuple[str, str]]:
    if not path.is_file():
        return []
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            t = (r.get("text") or "").strip().strip('"')
            e = (r.get("emotion") or "").strip().lower()
            if not t or e not in LABEL2ID:
                continue
            rows.append((t, e))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json", type=Path, default=DATASET_JSON)
    parser.add_argument("--augment-csv", type=Path, default=DEFAULT_AUGMENT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-per-class", type=int, default=1200, help="Cap IEMOCAP lines per emotion (balance)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    rows = load_rows_from_dataset_responses(args.dataset_json, args.max_per_class)
    rows.extend(load_augment_csv(args.augment_csv))
    if len(rows) < 50:
        raise SystemExit(
            f"Too few training rows ({len(rows)}). Ensure {args.dataset_json} exists and/or augment CSV."
        )

    texts = [r[0] for r in rows]
    labels = [LABEL2ID[r[1]] for r in rows]

    cnt = Counter(labels)
    stratify_arg = labels if len(cnt) > 1 and min(cnt.values()) >= 2 else None
    try:
        tx, vx, ty, vy = train_test_split(
            texts, labels, test_size=0.1, random_state=args.seed, stratify=stratify_arg
        )
    except ValueError:
        tx, vx, ty, vy = train_test_split(texts, labels, test_size=0.1, random_state=args.seed)

    ds = DatasetDict(
        {
            "train": Dataset.from_dict({"text": tx, "labels": ty}),
            "validation": Dataset.from_dict({"text": vx, "labels": vy}),
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tok(batch):
        enc = tokenizer(batch["text"], truncation=True, max_length=args.max_length, padding=False)
        enc["labels"] = batch["labels"]
        return enc

    ds = ds.map(tok, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    def compute_metrics(eval_pred):
        logits, labels_arr = eval_pred
        pred = np.argmax(logits, axis=-1)
        acc = float((pred == labels_arr).mean())
        return {"accuracy": acc}

    args_out = args.output
    args_out.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(args_out / "checkpoints"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["validation"],
            processing_class=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )
    except TypeError:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["validation"],
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )
    trainer.train()
    trainer.save_model(str(args_out))
    tokenizer.save_pretrained(str(args_out))

    meta = {
        "base_model": args.model,
        "labels": LABELS,
        "label2id": LABEL2ID,
        "train_rows": len(tx),
        "val_rows": len(vx),
        "sources": [str(args.dataset_json), str(args.augment_csv)],
    }
    with open(args_out / "text_emotion_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved text emotion model to {args_out.resolve()}")
    print("Restart the emotion API (uvicorn) to load it.")


if __name__ == "__main__":
    main()
