"""SFT finetuning of Gemma-3-27B-it (Section 4.1; Appendix E, F).

2 epochs, lr 1e-4, LoRA rank-64 (alpha 128) on all attention+MLP projections,
effective batch size 8. Trains on the calm + instruct-mix dataset built by
``build_sft``. The ``--variant teacher`` flag trains the Appendix F failure
model.

Usage:
    python -m emotional_instability.training.train_sft --variant diverse
    python -m emotional_instability.training.train_sft --variant teacher
"""
from __future__ import annotations

import argparse

from ..config import load_config
from .hyperparams import sft_from_config


def _grad_accum(effective_batch: int, per_device: int) -> int:
    return max(1, effective_batch // per_device)


def train_sft(config, variant: str = "diverse", per_device_batch: int = 1) -> str:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    hp = sft_from_config(config)
    spec = config.model_by_name(config.finetune_base)
    data_path = str(config.output_path("training", f"sft_{variant}.jsonl"))
    out_dir = str(config.output_path("checkpoints", f"sft_{variant}").parent / f"sft_{variant}")

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    peft_config = LoraConfig(
        r=hp.lora_rank, lora_alpha=hp.lora_alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=hp.target_modules,
    )

    ds = load_dataset("json", data_files=data_path, split="train")

    args = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=hp.epochs,
        learning_rate=hp.learning_rate,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=_grad_accum(hp.effective_batch_size, per_device_batch),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        gradient_checkpointing=True,
        # Train only on assistant completions (mask user/system tokens).
        assistant_only_loss=True,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=ds,
        processing_class=tokenizer, peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(out_dir)
    print(f"[sft:{variant}] adapter saved -> {out_dir}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="SFT finetune Gemma-3-27B-it")
    ap.add_argument("--config", default=None)
    ap.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    ap.add_argument("--per-device-batch", type=int, default=1)
    args = ap.parse_args()
    train_sft(load_config(args.config), args.variant, args.per_device_batch)


if __name__ == "__main__":
    main()
