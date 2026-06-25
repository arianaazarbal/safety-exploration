#!/usr/bin/env python
"""Section 4.1: train the DPO or SFT LoRA finetune of Gemma-3-27b-it.

Examples:
    python scripts/train_finetune.py dpo --data data/dpo_pairs.jsonl
    python scripts/train_finetune.py sft --data data/sft_dataset.jsonl
    # Appendix I layer ablation: adapters on layers 30-35 only
    python scripts/train_finetune.py dpo --data data/dpo_pairs.jsonl --layers 30 31 32 33 34
"""

from __future__ import annotations

import argparse
import os

import _bootstrap  # noqa: F401  (puts repo root on sys.path)

from emotional_instability import config
from emotional_instability.training import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["dpo", "sft"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=config.TARGET_FINETUNE_MODEL)
    ap.add_argument("--layers", nargs="*", type=int, default=None,
                    help="restrict LoRA adapters to these decoder layers (Appendix I)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or os.path.join(config.CHECKPOINTS_DIR, f"gemma27b_{args.method}")
    if args.layers:
        out += "_layers" + "-".join(map(str, args.layers))

    if args.method == "dpo":
        train.train_dpo(args.data, out, model_name=args.model, layers=args.layers)
    else:
        train.train_sft(args.data, out, model_name=args.model, layers=args.layers)
    print(f"Saved adapter to {out}")


if __name__ == "__main__":
    main()
