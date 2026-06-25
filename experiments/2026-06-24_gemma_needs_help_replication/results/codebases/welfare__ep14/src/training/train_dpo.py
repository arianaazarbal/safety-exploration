"""DPO finetuning of Gemma-3-27B-it (Section 4, Appendix E).

LoRA rank-64 adapters on all attention + MLP projections; 1 epoch; lr 5e-5;
beta 0.1; effective batch size 8; 280 preference pairs.

Usage:
    python -m src.training.train_dpo \
        --data data/dpo_pairs.jsonl --out checkpoints/dpo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import config


def load_pairs(path: str):
    from datasets import Dataset
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(config.DATA_DIR / "dpo_pairs.jsonl"))
    ap.add_argument("--out", default=str(config.MODELS_DIR / "dpo"))
    ap.add_argument("--base", default=config.FINETUNE_BASE.model_id)
    ap.add_argument("--layers", default=None,
                    help="optional 'start-end' to restrict LoRA layers (Appendix I)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import DPOConfig as TRLDPOConfig, DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto",
    )

    target_modules = config.LORA_TARGET_MODULES
    layers_to_transform = None
    if args.layers:
        lo, hi = (int(x) for x in args.layers.split("-"))
        layers_to_transform = list(range(lo, hi))

    peft_config = LoraConfig(
        r=config.DPO.lora_rank,
        lora_alpha=config.DPO.lora_alpha,
        target_modules=target_modules,
        layers_to_transform=layers_to_transform,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = load_pairs(args.data)

    # Effective batch size 8 via per-device batch * grad accumulation.
    per_device = 1
    grad_accum = max(1, config.DPO.effective_batch_size // per_device)

    training_args = TRLDPOConfig(
        output_dir=args.out,
        num_train_epochs=config.DPO.epochs,
        learning_rate=config.DPO.learning_rate,
        beta=config.DPO.beta,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        seed=config.SEED,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved DPO adapter -> {args.out}")


if __name__ == "__main__":
    main()
