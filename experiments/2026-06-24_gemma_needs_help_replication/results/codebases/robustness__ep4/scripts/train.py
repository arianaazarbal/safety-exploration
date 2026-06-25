#!/usr/bin/env python
"""Train the SFT or DPO mitigation on Gemma-3-27B-it (Section 4.1).

Examples
--------
python scripts/train.py dpo --dataset outputs/data/dpo.jsonl --out outputs/models/gemma-dpo
python scripts/train.py sft --dataset outputs/data/sft.jsonl --out outputs/models/gemma-sft
# Section 4.2 ablation: restrict LoRA to central layers
python scripts/train.py dpo --dataset outputs/data/dpo.jsonl \
    --out outputs/models/gemma-dpo-l30-35 --layers 30 31 32 33 34 35
"""
from __future__ import annotations

import argparse

import _common  # noqa: F401

from instability.training.dpo import DPOConfig, train_dpo
from instability.training.sft import SFTConfig, train_sft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["sft", "dpo"])
    ap.add_argument("--base-model", default="google/gemma-3-27b-it")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--layers", nargs="+", type=int, default=None,
                    help="(DPO) restrict LoRA to these decoder layers")
    args = ap.parse_args()

    if args.method == "sft":
        cfg = SFTConfig(base_model=args.base_model, dataset_path=args.dataset,
                        output_dir=args.out, load_in_4bit=args.load_in_4bit)
        train_sft(cfg)
    else:
        cfg = DPOConfig(base_model=args.base_model, dataset_path=args.dataset,
                        output_dir=args.out, load_in_4bit=args.load_in_4bit,
                        layers=args.layers)
        train_dpo(cfg)


if __name__ == "__main__":
    main()
