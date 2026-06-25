#!/usr/bin/env python
"""Train the DPO or SFT LoRA adapter on Gemma-3-27B-it (Section 4.1).

Usage:
    python scripts/train.py --method dpo --data data/finetune/dpo_pairs.jsonl \
        --output-dir runs/dpo
    python scripts/train.py --method sft --data data/finetune/sft.jsonl \
        --output-dir runs/sft
    # Appendix I layer ablation (train only layers 30-35):
    python scripts/train.py --method dpo --data data/finetune/dpo_pairs.jsonl \
        --output-dir runs/dpo_l30_35 --layers 30 31 32 33 34 35
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from emotional_instability.finetune.train import dpo_config, sft_config, train_dpo, train_sft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["dpo", "sft"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--base-model", default="google/gemma-3-27b-it")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--layers", nargs="*", type=int, default=None,
                    help="restrict LoRA to these layer indices (Appendix I ablation)")
    args = ap.parse_args()

    common = dict(
        base_model=args.base_model,
        output_dir=args.output_dir,
        load_in_4bit=not args.no_4bit,
        layers_to_transform=args.layers,
    )
    if args.method == "dpo":
        cfg = dpo_config(**common)
        train_dpo(cfg, args.data)
    else:
        cfg = sft_config(**common)
        train_sft(cfg, args.data)
    print(f"saved adapter -> {args.output_dir}")


if __name__ == "__main__":
    main()
