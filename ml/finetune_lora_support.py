#!/usr/bin/env python3
"""
LoRA fine-tune a small instruct model on empathetic support dialogue (JSONL).

Goal: better generalization / style on unseen user messages—not memorizing IEMOCAP lines.
After training, merge or serve the adapter (Ollama import, vLLM, HF transformers).

Example (GPU ~8GB+ recommended; 0.5B models work on smaller VRAM):
  pip install -r requirements.txt -r requirements-train.txt
  python ml/finetune_lora_support.py --model Qwen/Qwen2.5-0.5B-Instruct --epochs 3

CPU: possible but very slow; use --epochs 1 and tiny model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def load_jsonl_messages(path: Path) -> Dataset:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "messages" not in obj or not isinstance(obj["messages"], list):
                raise ValueError(f"Each line needs a 'messages' array: {path}")
            rows.append(obj)
    if not rows:
        raise ValueError(f"No examples in {path}")
    return Dataset.from_list(rows)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="LoRA SFT for mental-health support style")
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "data" / "empathic_support_train.jsonl",
        help="JSONL with {\"messages\": [...]} per line (chat roles)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HF model id (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0, Qwen/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "models" / "lora_empathic_support",
        help="Directory to save LoRA adapter",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=2, help="Per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument(
        "--4bit",
        action="store_true",
        help="Load base model in 4-bit (needs bitsandbytes + GPU)",
    )
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"Data file not found: {args.data}")

    args.output.mkdir(parents=True, exist_ok=True)

    dataset = load_jsonl_messages(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_text(example: dict) -> dict:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(format_text, remove_columns=["messages"])

    bnb_config = None
    if args.__dict__["4bit"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=dtype if bnb_config is None else None,
    )

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
    )

    # TRL >= 1.0: dataset_text_field, max_length, packing live on SFTConfig—not on SFTTrainer.
    sft_args = SFTConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=5,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="none",
        gradient_checkpointing=False,
        optim="adamw_torch",
        dataset_text_field="text",
        max_length=args.max_seq_len,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))

    meta = {
        "base_model": args.model,
        "train_examples": len(dataset),
        "data_file": str(args.data.resolve()),
    }
    with open(args.output / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. Adapter saved to {args.output.resolve()}")
    print("Next: merge for Ollama/vLLM, or point an OpenAI-compatible server at this adapter + base weights.")


if __name__ == "__main__":
    main()
