#!/usr/bin/env python3
"""
Evaluate the trained text emotion classifier and export thesis-ready evidence.

Outputs:
  ml/eval_outputs/text_emotion/<timestamp>/
    - metrics.json
    - classification_report.txt
    - predictions.csv
    - confusion_matrix.png
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = ROOT / "models" / "text_emotion"
DEFAULT_DATASET_JSON = ROOT / "models" / "dataset_responses.json"
DEFAULT_AUGMENT_CSV = ROOT / "data" / "text_emotion_augment.csv"
DEFAULT_OUT_ROOT = ROOT / "eval_outputs" / "text_emotion"


def load_rows_from_dataset_responses(path: Path) -> list[tuple[str, str]]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows: list[tuple[str, str]] = []
    if not isinstance(data, dict):
        return rows
    for emo, pairs in data.items():
        if not isinstance(pairs, list):
            continue
        for p in pairs:
            if not isinstance(p, dict):
                continue
            u = (p.get("user") or "").strip()
            if len(u) < 3:
                continue
            rows.append((u, str(emo).strip().lower()))
    return rows


def load_augment_csv(path: Path) -> list[tuple[str, str]]:
    if not path.is_file():
        return []
    rows: list[tuple[str, str]] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            t = (r.get("text") or "").strip().strip('"')
            e = (r.get("emotion") or "").strip().lower()
            if not t or not e:
                continue
            rows.append((t, e))
    return rows


def batched(iterable, n: int):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def save_confusion_matrix_png(cm: np.ndarray, labels: list[str], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_w = max(7, int(len(labels) * 0.8))
    fig_h = max(6, int(len(labels) * 0.7))
    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix (counts)")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = int(cm[i, j])
            if v == 0:
                continue
            plt.text(j, i, str(v), ha="center", va="center", color="white" if v > thresh else "black", fontsize=8)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def make_stratified_split_with_min_test(
    texts: list[str],
    y: np.ndarray,
    test_size: float,
    seed: int,
    min_test_per_class: int,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, dict[str, int]]:
    """
    Create a split that tries to ensure at least `min_test_per_class` examples per class in the test set.
    If a class has fewer than `min_test_per_class` total examples, it is marked as "rare".
    """
    rng = np.random.default_rng(seed)
    n = len(texts)
    idx_by_class: dict[int, list[int]] = {}
    for i, c in enumerate(y.tolist()):
        idx_by_class.setdefault(int(c), []).append(i)
    for c in idx_by_class:
        rng.shuffle(idx_by_class[c])

    total_test = max(1, int(round(n * test_size)))
    test_idx: list[int] = []
    rare: dict[str, int] = {}

    # First, try to allocate min per class (only if available)
    for c, idxs in idx_by_class.items():
        take = min(min_test_per_class, len(idxs))
        if take < min_test_per_class:
            rare[str(c)] = len(idxs)
        test_idx.extend(idxs[:take])

    # Then fill remaining test slots proportionally from remaining pool
    test_idx = list(dict.fromkeys(test_idx))  # de-dup, preserve order
    remaining_pool = [i for i in range(n) if i not in set(test_idx)]
    need = max(0, total_test - len(test_idx))
    if need > 0 and remaining_pool:
        pick = rng.choice(np.array(remaining_pool, dtype=np.int64), size=min(need, len(remaining_pool)), replace=False)
        test_idx.extend([int(i) for i in pick.tolist()])

    test_set = set(test_idx)
    train_idx = [i for i in range(n) if i not in test_set]

    x_train = [texts[i] for i in train_idx]
    x_test = [texts[i] for i in test_idx]
    y_train = y[np.array(train_idx, dtype=np.int64)]
    y_test = y[np.array(test_idx, dtype=np.int64)]
    return x_train, x_test, y_train, y_test, rare


def compute_metrics_block(
    y_true: list[int],
    y_pred: list[int],
    labels: list[str],
    include_label_ids: list[int] | None = None,
) -> dict:
    arr_true = np.array(y_true, dtype=np.int64)
    arr_pred = np.array(y_pred, dtype=np.int64)
    acc = float(np.mean(arr_true == arr_pred))

    if include_label_ids is None:
        include_label_ids = list(range(len(labels)))

    macro_f1 = float(f1_score(arr_true, arr_pred, labels=include_label_ids, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(arr_true, arr_pred, labels=include_label_ids, average="weighted", zero_division=0))
    precision_macro, recall_macro, _, _ = precision_recall_fscore_support(
        arr_true, arr_pred, labels=include_label_ids, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "included_labels": [labels[i] for i in include_label_ids],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET_JSON)
    parser.add_argument("--augment-csv", type=Path, default=DEFAULT_AUGMENT_CSV)
    parser.add_argument("--extra-csv", type=Path, action="append", default=[], help="Additional CSVs with columns text,emotion")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument(
        "--min-test-per-class",
        type=int,
        default=10,
        help="Try to ensure at least this many examples per emotion in test. Very rare classes will be reported separately.",
    )
    parser.add_argument(
        "--macro-min-support",
        type=int,
        default=10,
        help="Compute additional 'macro (supported classes only)' metrics using classes with at least this many test examples.",
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = parser.parse_args()

    meta_path = args.model_dir / "text_emotion_meta.json"
    if not meta_path.is_file():
        raise SystemExit(f"Missing model meta: {meta_path}. Train first: python ml/train_text_emotion.py")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    labels: list[str] = list(meta.get("labels") or [])
    label2id: dict[str, int] = dict(meta.get("label2id") or {})
    if not labels or not label2id:
        raise SystemExit(f"Invalid labels in meta: {meta_path}")

    rows = []
    rows.extend(load_rows_from_dataset_responses(args.dataset_json))
    rows.extend(load_augment_csv(args.augment_csv))
    for p in args.extra_csv or []:
        rows.extend(load_augment_csv(p))

    rows = [(t, e) for (t, e) in rows if e in label2id]
    if len(rows) < 200:
        raise SystemExit(f"Too few labeled rows for evaluation: {len(rows)}")

    texts = [t for (t, _) in rows]
    y = np.array([label2id[e] for (_, e) in rows], dtype=np.int64)

    x_train, x_test, y_train, y_test, rare_info = make_stratified_split_with_min_test(
        texts=texts,
        y=y,
        test_size=args.test_size,
        seed=args.seed,
        min_test_per_class=args.min_test_per_class,
    )

    # Load model and run inference
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    preds: list[int] = []
    probs_max: list[float] = []
    with torch.no_grad():
        for batch_texts in batched(x_test, args.batch):
            enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            p = torch.softmax(logits, dim=-1)
            pred = torch.argmax(p, dim=-1).cpu().numpy().tolist()
            mx = torch.max(p, dim=-1).values.cpu().numpy().tolist()
            preds.extend(pred)
            probs_max.extend([float(v) for v in mx])

    y_true = y_test.astype(int).tolist()
    y_pred = [int(v) for v in preds]

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save report text
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        digits=4,
        zero_division=0,
    )
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    # Baseline: TF-IDF + Logistic Regression
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000)
    Xtr = tfidf.fit_transform(x_train)
    Xte = tfidf.transform(x_test)
    lr = LogisticRegression(max_iter=2000, n_jobs=1, class_weight="balanced")
    lr.fit(Xtr, y_train)
    base_pred = lr.predict(Xte).astype(int).tolist()

    # Save predictions CSV (for evidence + examples)
    with open(out_dir / "predictions.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "true_label", "model_pred", "model_confidence", "baseline_pred"])
        for t, yt, yp, conf, bp in zip(x_test, y_true, y_pred, probs_max, base_pred):
            w.writerow([t, labels[yt], labels[yp], f"{conf:.6f}", labels[int(bp)]])

    # Save confusion matrix plot
    save_confusion_matrix_png(cm, labels, out_dir / "confusion_matrix.png")

    # Support-aware macro metrics: compute macro over classes with enough test support
    test_support = Counter(y_true)
    supported_label_ids = sorted([i for i, c in test_support.items() if c >= args.macro_min_support])
    if not supported_label_ids:
        supported_label_ids = list(range(len(labels)))

    # Save metrics JSON (easy to cite in thesis)
    metrics_model_all = compute_metrics_block(y_true, y_pred, labels, include_label_ids=list(range(len(labels))))
    metrics_base_all = compute_metrics_block(y_true, base_pred, labels, include_label_ids=list(range(len(labels))))
    metrics_model_supported = compute_metrics_block(y_true, y_pred, labels, include_label_ids=supported_label_ids)
    metrics_base_supported = compute_metrics_block(y_true, base_pred, labels, include_label_ids=supported_label_ids)

    metrics = {
        "model_dir": str(args.model_dir),
        "dataset_json": str(args.dataset_json),
        "augment_csv": str(args.augment_csv),
        "extra_csv": [str(p) for p in (args.extra_csv or [])],
        "labels": labels,
        "n_total_rows": len(rows),
        "n_test": len(x_test),
        "test_size": args.test_size,
        "seed": args.seed,
        "device": str(device),
        "split": {
            "min_test_per_class": args.min_test_per_class,
            "rare_classes_total_counts": {labels[int(k)]: int(v) for k, v in (rare_info or {}).items()},
            "test_support_per_class": {labels[int(k)]: int(v) for k, v in Counter(y_true).items()},
        },
        "metrics": {
            "model_all_classes": metrics_model_all,
            "baseline_all_classes": metrics_base_all,
            "model_supported_classes_only": metrics_model_supported,
            "baseline_supported_classes_only": metrics_base_supported,
        },
        "delta_supported_macro_f1": float(
            metrics_model_supported["macro_f1"] - metrics_base_supported["macro_f1"]
        ),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved evaluation to:", out_dir.resolve())
    print(
        json.dumps(
            {
                "model_supported_macro_f1": metrics_model_supported["macro_f1"],
                "baseline_supported_macro_f1": metrics_base_supported["macro_f1"],
                "delta_supported_macro_f1": metrics["delta_supported_macro_f1"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

