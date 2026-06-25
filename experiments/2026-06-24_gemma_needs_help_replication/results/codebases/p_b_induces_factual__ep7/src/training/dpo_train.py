"""DPO fine-tuning of Gemma-3-27B-it on the 280 preference pairs (Section 4 / Table 9).

Hyperparameters (paper): 1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 / alpha 64 on all
attention+MLP projections, effective batch size 8. The trained LoRA adapter is saved to
``CHECKPOINT_DIR/<adapter_name>`` (default ``dpo``) so the eval harness can target it as
``gemma-3-27b-it+dpo``.

Usage:
    python -m src.training.dpo_train
    python -m src.training.dpo_train --layer-ablation layers_30_35 --adapter-name dpo_l30_35
"""
from __future__ import annotations

import argparse
from pathlib import Path

import config
from .lora_common import load_base_for_training, make_lora_config


def train_dpo(
    *,
    dataset_path: Path,
    adapter_name: str,
    base_model_id: str,
    layer_ablation: str,
    load_in_4bit: bool,
):
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    model, tokenizer = load_base_for_training(base_model_id, load_in_4bit=load_in_4bit)
    peft_config = make_lora_config(
        rank=config.DPO.lora_rank, alpha=config.DPO.lora_alpha, layer_ablation=layer_ablation
    )

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    grad_accum = max(1, config.DPO.effective_batch_size // config.DPO.per_device_batch_size)
    out_dir = config.CHECKPOINT_DIR / adapter_name

    args = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=config.DPO.epochs,
        learning_rate=config.DPO.learning_rate,
        beta=config.DPO.beta,
        per_device_train_batch_size=config.DPO.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=config.DPO.max_length,
        max_prompt_length=config.DPO.max_prompt_length,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,            # peft: ref = base model with adapter disabled
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"[dpo_train] saved adapter -> {out_dir}  (eval as: gemma-3-27b-it+{adapter_name})")


def main():
    ap = argparse.ArgumentParser(description="DPO fine-tuning (Section 4)")
    ap.add_argument("--dataset", default=str(config.DATA_DIR / "dpo_pairs.jsonl"))
    ap.add_argument("--adapter-name", default="dpo")
    ap.add_argument("--base-model", default=config.TARGET_MODELS["gemma-3-27b-it"].model_id)
    ap.add_argument("--layer-ablation", default="all", choices=list(config.LORA_LAYER_ABLATIONS))
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()
    train_dpo(
        dataset_path=Path(args.dataset),
        adapter_name=args.adapter_name,
        base_model_id=args.base_model,
        layer_ablation=args.layer_ablation,
        load_in_4bit=not args.no_4bit,
    )


if __name__ == "__main__":
    main()
