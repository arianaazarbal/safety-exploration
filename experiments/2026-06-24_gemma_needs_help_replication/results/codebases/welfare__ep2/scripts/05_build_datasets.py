#!/usr/bin/env python
"""Section 4.1: build the DPO preference pairs and SFT dataset.

Requires a scored gemma-3-27b-it eval run (scripts/01) and a calm pool
(scripts/04).

    python scripts/05_build_datasets.py --dpo --sft
"""
import argparse

import _bootstrap  # noqa: F401
from gemma_distress.training.build_datasets import build_dpo_dataset, build_sft_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo", action="store_true")
    ap.add_argument("--sft", action="store_true")
    args = ap.parse_args()
    if not (args.dpo or args.sft):
        args.dpo = args.sft = True

    if args.dpo:
        pairs = build_dpo_dataset()
        print(f"DPO: built {len(pairs)} preference pairs.")
    if args.sft:
        ex = build_sft_dataset()
        print(f"SFT: built {len(ex)} examples (calm + instruct mix).")


if __name__ == "__main__":
    main()
