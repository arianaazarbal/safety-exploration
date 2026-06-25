#!/usr/bin/env python
"""Section 4.1 / Appendix E/I: LoRA finetune Gemma-3-27B-it (DPO or SFT).

Usage:
    python scripts/08_train.py --method dpo --data outputs/training/dpo_pairs.jsonl \\
        --out outputs/training/dpo
    python scripts/08_train.py --method sft --data outputs/training/sft_data.jsonl \\
        --out outputs/training/sft_diverse
    # Appendix I layer ablation:
    python scripts/08_train.py --method dpo --data ... --out ... --layers 30 35
"""

from __future__ import annotations

import argparse
import dataclasses

from _common import load
from gemma_distress.training import config as tconfig
from gemma_distress.training.train import train_dpo, train_sft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["dpo", "sft"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-model", default="google/gemma-3-27b-it")
    ap.add_argument("--layers", nargs=2, type=int, default=None,
                    metavar=("LO", "HI"),
                    help="restrict LoRA to decoder layers [LO, HI) (Appendix I)")
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    args = ap.parse_args()

    load()  # validates config presence
    tc = tconfig.DPO if args.method == "dpo" else tconfig.SFT
    if args.layers:
        tc = dataclasses.replace(tc, layer_subset=(args.layers[0], args.layers[1]))

    fn = train_dpo if args.method == "dpo" else train_sft
    adapter = fn(args.base_model, args.data, args.out, tc,
                 per_device_batch_size=args.per_device_batch_size)
    print(f"Saved adapter -> {adapter}")


if __name__ == "__main__":
    main()
