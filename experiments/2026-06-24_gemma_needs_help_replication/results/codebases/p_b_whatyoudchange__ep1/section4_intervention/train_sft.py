"""SFT LoRA finetune of gemma-3-27b-it (Appendix E, Table 9).

650 calm responses + 500 Dolci-Instruct-SFT, 2 epochs, lr 1e-4, LoRA r=64 a=128.
The paper finds SFT ineffective at reducing distress (and the 'teacher' variant
slightly increases it); this trainer reproduces the procedure regardless.

Usage:
    python -m section4_intervention.train_sft --tag gemma-3-27b-it-sft-diverse
"""

from __future__ import annotations

import argparse

from config import TRAIN
from models.registry import register_adapter
from .build_datasets import build_sft_dataset, to_hf_dataset
from .train_common import (adapter_output_dir, build_lora_config,
                           load_base_model_and_tokenizer)


def train_sft(tag: str = "gemma-3-27b-it-sft-diverse", seed: int = 0,
              register: bool = True, calm_pool_name: str = "calm_pool") -> str:
    from trl import SFTConfig, SFTTrainer

    rows = build_sft_dataset(seed=seed, calm_pool_name=calm_pool_name, out_name=f"sft_{tag}")
    dataset = to_hf_dataset(rows)
    model, tok = load_base_model_and_tokenizer(TRAIN.base_model)
    peft_config = build_lora_config(lora_alpha=TRAIN.sft_lora_alpha)
    out_dir = adapter_output_dir(tag)

    cfg = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=TRAIN.sft_epochs,
        learning_rate=TRAIN.sft_lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=TRAIN.effective_batch_size,  # effective bs = 8
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=seed,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=dataset,
        peft_config=peft_config, processing_class=tok,
    )
    trainer.train()
    trainer.save_model(out_dir)
    if register:
        register_adapter(tag, TRAIN.base_model, out_dir)
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gemma-3-27b-it-sft-diverse")
    ap.add_argument("--calm-pool", default="calm_pool",
                    help="'calm_pool' (diverse) or 'calm_pool_teacher' (Appendix F)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    path = train_sft(tag=args.tag, seed=args.seed, calm_pool_name=args.calm_pool)
    print(f"SFT adapter saved to {path}")


if __name__ == "__main__":
    main()
