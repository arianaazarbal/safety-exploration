#!/usr/bin/env python
"""Section 4.1: train the DPO / SFT LoRA finetune of Gemma-3-27B-it.

Examples
--------
python scripts/05_train.py --method dpo --dataset outputs/training/dpo_dataset.jsonl
python scripts/05_train.py --method sft --dataset outputs/training/sft_dataset.jsonl

# Appendix I layer-subset ablation (adapters on layers 30-34 only):
python scripts/05_train.py --method dpo --dataset outputs/training/dpo_dataset.jsonl \
    --lora-layers 30 31 32 33 34 --output-dir outputs/training/dpo_l30_35
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability.config import load_training_config  # noqa: E402
from emotional_instability.training.train import train_dpo, train_sft  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["dpo", "sft"], required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--lora-layers", nargs="+", type=int, default=None,
                    help="restrict LoRA adapters to these decoder layer indices (Appendix I)")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = load_training_config()
    if args.lora_layers is not None:
        cfg[args.method]["lora_layers"] = args.lora_layers

    if args.method == "dpo":
        train_dpo(args.dataset, training_cfg=cfg, per_device_batch=args.per_device_batch,
                  output_dir=args.output_dir)
    else:
        train_sft(args.dataset, training_cfg=cfg, per_device_batch=args.per_device_batch,
                  output_dir=args.output_dir)


if __name__ == "__main__":
    main()
