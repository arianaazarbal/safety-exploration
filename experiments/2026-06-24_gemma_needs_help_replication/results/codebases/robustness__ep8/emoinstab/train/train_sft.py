"""SFT finetuning of Gemma-3-27B-it (Section 4.1, Appendix E).

Hyperparameters (Table 9): 1,150 samples (650 calm + 500 instruct mix), 2 epochs,
lr 1e-4, LoRA rank 64 / alpha 128, effective batch size 8. The paper finds SFT
ineffective (and the 'teacher' variant counter-productive) — this trainer
reproduces that negative result.

    python -m emoinstab.train.train_sft \
        --dataset outputs/datasets/sft.jsonl \
        --output outputs/checkpoints/sft
"""
from __future__ import annotations

import argparse

from emoinstab.config import LoRAConfig, SFTTrainConfig
from emoinstab.train.lora import build_lora_config


def train(cfg: SFTTrainConfig, dataset_path: str):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    ds = load_dataset("json", data_files=dataset_path, split="train")

    grad_accum = max(1, cfg.effective_batch_size // cfg.per_device_batch_size)
    sft_args = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=cfg.max_length,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        # Conversational dataset uses the "messages" column + chat template.
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=build_lora_config(cfg.lora),
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"Saved SFT adapter to {cfg.output_dir}")


def main():
    ap = argparse.ArgumentParser(description="SFT finetuning (Section 4.1).")
    ap.add_argument("--dataset", default="outputs/datasets/sft.jsonl")
    ap.add_argument("--base-model", default="google/gemma-3-27b-it")
    ap.add_argument("--output", default="outputs/checkpoints/sft")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    args = ap.parse_args()

    cfg = SFTTrainConfig(
        base_model=args.base_model,
        epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output,
        lora=LoRAConfig(rank=args.lora_rank, alpha=args.lora_alpha),
    )
    train(cfg, args.dataset)


if __name__ == "__main__":
    main()
