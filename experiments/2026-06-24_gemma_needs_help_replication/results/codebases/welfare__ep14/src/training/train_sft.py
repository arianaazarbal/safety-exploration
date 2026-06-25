"""SFT finetuning of Gemma-3-27B-it (Section 4, Appendix E).

LoRA rank-64 (alpha 128) on all projections; 2 epochs; lr 1e-4; effective batch
size 8; 1,150 samples (650 calm + 500 instruct mix). Per the paper this baseline
fails to reduce frustration -- it is included to reproduce that negative result.
Pass --teacher to train on the 'teacher' calm dataset variant (Appendix F).

Usage:
    python -m src.training.train_sft --data data/sft_samples.jsonl --out checkpoints/sft
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import config


def load_samples(path: str):
    from datasets import Dataset
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(config.DATA_DIR / "sft_samples.jsonl"))
    ap.add_argument("--out", default=str(config.MODELS_DIR / "sft"))
    ap.add_argument("--base", default=config.FINETUNE_BASE.model_id)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto",
    )

    peft_config = LoraConfig(
        r=config.SFT.lora_rank,
        lora_alpha=config.SFT.lora_alpha,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = load_samples(args.data)
    per_device = 1
    grad_accum = max(1, config.SFT.effective_batch_size // per_device)

    training_args = SFTConfig(
        output_dir=args.out,
        num_train_epochs=config.SFT.epochs,
        learning_rate=config.SFT.learning_rate,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        seed=config.SEED,
        report_to=[],
        max_seq_length=4096,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved SFT adapter -> {args.out}")


if __name__ == "__main__":
    main()
