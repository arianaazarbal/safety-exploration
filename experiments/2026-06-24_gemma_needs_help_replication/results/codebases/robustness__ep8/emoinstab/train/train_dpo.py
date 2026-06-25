"""DPO finetuning of Gemma-3-27B-it (Section 4.1, Appendix E).

Hyperparameters (Table 9): 280 pairs, 1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 /
alpha 64, effective batch size 8. Trains on conversational preference data built
by ``build_datasets.build_dpo_dataset``.

    python -m emoinstab.train.train_dpo \
        --dataset outputs/datasets/dpo.jsonl \
        --output outputs/checkpoints/dpo
"""
from __future__ import annotations

import argparse

from emoinstab.config import DPOTrainConfig, LoRAConfig
from emoinstab.train.lora import build_lora_config


def train(cfg: DPOTrainConfig, dataset_path: str):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    ds = load_dataset("json", data_files=dataset_path, split="train")

    grad_accum = max(1, cfg.effective_batch_size // cfg.per_device_batch_size)
    dpo_args = DPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=build_lora_config(cfg.lora),
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"Saved DPO adapter to {cfg.output_dir}")


def main():
    ap = argparse.ArgumentParser(description="DPO finetuning (Section 4.1).")
    ap.add_argument("--dataset", default="outputs/datasets/dpo.jsonl")
    ap.add_argument("--base-model", default="google/gemma-3-27b-it")
    ap.add_argument("--output", default="outputs/checkpoints/dpo")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--layers", default=None,
                    help="optional layer band 'lo-hi' for Appendix I ablation, e.g. 30-35")
    args = ap.parse_args()

    layers = None
    if args.layers:
        lo, hi = args.layers.split("-")
        layers = (int(lo), int(hi))
    cfg = DPOTrainConfig(
        base_model=args.base_model,
        epochs=args.epochs,
        learning_rate=args.lr,
        beta=args.beta,
        output_dir=args.output,
        lora=LoRAConfig(rank=args.lora_rank, alpha=args.lora_alpha, layers_to_transform=layers),
    )
    train(cfg, args.dataset)


if __name__ == "__main__":
    main()
