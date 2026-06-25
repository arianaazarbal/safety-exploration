"""SFT finetuning of Gemma-3-27B-it (Section 4.1 / Appendix E).

Hyperparameters (Table 9):
  dataset = 1,150 samples (650 calm + 500 Dolci), epochs = 2, lr = 1e-4,
  LoRA rank = 64, alpha = 128, effective batch size = 8,
  LoRA on all attention + MLP projections.

This is the "diverse" SFT variant. The "teacher" variant (Appendix F) is
produced by regenerating calm data with the teacher system prompt; pass that
dataset via --dataset to train it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from emotional_instability.config import ARTIFACTS_DIR, GLOBAL_SEED, TARGET_MODELS
from finetuning.train_dpo import LORA_TARGET_MODULES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=ARTIFACTS_DIR / "sft_dataset.jsonl")
    ap.add_argument("--output", type=Path, default=ARTIFACTS_DIR / "gemma-3-27b-it-sft")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    spec = TARGET_MODELS["gemma-3-27b-it"]
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    ds = load_dataset("json", data_files=str(args.dataset), split="train")

    def _format(row):
        return {"text": tokenizer.apply_chat_template(row["messages"], tokenize=False)}

    ds = ds.map(_format)

    peft_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES,
    )

    cfg = SFTConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        seed=args.seed,
        dataset_text_field="text",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=ds,
        processing_class=tokenizer, peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(args.output))
    print(json.dumps({"adapter_path": str(args.output)}))


if __name__ == "__main__":
    main()
