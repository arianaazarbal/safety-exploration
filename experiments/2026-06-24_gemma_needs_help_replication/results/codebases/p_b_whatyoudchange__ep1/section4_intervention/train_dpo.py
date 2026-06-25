"""DPO LoRA finetune of gemma-3-27b-it (Appendix E, Table 9).

280 preference pairs (frustrated >= 3 vs matched calm), 1 epoch, lr 5e-5,
beta 0.1, LoRA r=64 a=64. This is the paper's headline mitigation: avg %>=5
drops 35% -> 0.3%.

Supports the §4.2 layer ablation via config.TRAIN.lora.layers_to_transform
(e.g. (30, 35) only, or (40, 61) only).

Usage:
    python -m section4_intervention.train_dpo --tag gemma-3-27b-it-dpo
"""

from __future__ import annotations

import argparse

from config import TRAIN
from models.registry import register_adapter
from .build_datasets import build_dpo_dataset, to_hf_dataset
from .train_common import (adapter_output_dir, build_lora_config,
                           load_base_model_and_tokenizer)


def train_dpo(tag: str = "gemma-3-27b-it-dpo", seed: int = 0,
              register: bool = True) -> str:
    from trl import DPOConfig, DPOTrainer

    rows = build_dpo_dataset(seed=seed)
    dataset = to_hf_dataset(rows)
    model, tok = load_base_model_and_tokenizer(TRAIN.base_model)
    peft_config = build_lora_config(lora_alpha=TRAIN.dpo_lora_alpha)
    out_dir = adapter_output_dir(tag)

    cfg = DPOConfig(
        output_dir=out_dir,
        num_train_epochs=TRAIN.dpo_epochs,
        learning_rate=TRAIN.dpo_lr,
        beta=TRAIN.dpo_beta,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=TRAIN.effective_batch_size,  # effective bs = 8
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=seed,
        report_to=[],
    )
    trainer = DPOTrainer(
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
    ap.add_argument("--tag", default="gemma-3-27b-it-dpo")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    path = train_dpo(tag=args.tag, seed=args.seed)
    print(f"DPO adapter saved to {path}")


if __name__ == "__main__":
    main()
