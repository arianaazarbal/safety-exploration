#!/usr/bin/env python
"""Section 4.1 / Appendix E/I: finetune Gemma-3-27B-it (DPO / SFT / layer ablation).

Example:
    python scripts/run_training.py --method dpo
    python scripts/run_training.py --method sft --flavour diverse
    python scripts/run_training.py --method dpo-layers --ranges 30-35 40-50
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distress_eval.training import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["dpo", "sft", "dpo-layers"])
    ap.add_argument("--flavour", default="diverse", choices=["diverse", "teacher"])
    ap.add_argument("--ranges", nargs="*", default=["30-35"],
                    help="layer ranges for dpo-layers, e.g. 30-35 40-50")
    args = ap.parse_args()

    if args.method == "dpo":
        train.train_dpo()
    elif args.method == "sft":
        train.train_sft(flavour=args.flavour)
    else:
        ranges = [tuple(int(x) for x in r.split("-")) for r in args.ranges]
        train.train_dpo_layer_ablation(ranges)


if __name__ == "__main__":
    main()
