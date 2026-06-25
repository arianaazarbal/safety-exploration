#!/usr/bin/env python
"""Section 4: train a LoRA finetune of Gemma-3-27B-it (DPO or SFT).

Profiles are defined in configs/finetune.yaml: dpo, sft, sft_teacher,
dpo_layers_30_35 (Appendix I ablation).

Example:
    python scripts/05_train.py --profile dpo
    python scripts/05_train.py --profile sft
"""
import _bootstrap  # noqa: F401
import argparse

from distress.config import FinetuneConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="dpo")
    ap.add_argument("--config", default="configs/finetune.yaml")
    args = ap.parse_args()

    cfg = FinetuneConfig.load(args.config, profile=args.profile)
    if cfg.method == "dpo":
        from distress.finetune.train_dpo import train
    else:
        from distress.finetune.train_sft import train
    train(cfg)


if __name__ == "__main__":
    main()
