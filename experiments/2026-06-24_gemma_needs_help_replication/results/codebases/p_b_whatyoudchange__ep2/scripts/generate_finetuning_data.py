#!/usr/bin/env python
"""Section 4.1: generate calm data, then build the DPO and SFT datasets.

Example:
    python scripts/generate_finetuning_data.py --all
    python scripts/generate_finetuning_data.py --calm diverse teacher
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distress_eval.training import build_datasets, calm_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calm", nargs="*", default=["diverse"],
                    choices=["diverse", "teacher"])
    ap.add_argument("--n-conversations", type=int, default=400)
    ap.add_argument("--build-dpo", action="store_true")
    ap.add_argument("--build-sft", nargs="*", choices=["diverse", "teacher"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    calm_flavours = ["diverse", "teacher"] if args.all else args.calm
    for flavour in calm_flavours:
        calm_data.generate(flavour=flavour, n_conversations=args.n_conversations, seed=args.seed)

    if args.all or args.build_dpo:
        build_datasets.build_dpo(seed=args.seed)
    sft_flavours = ["diverse", "teacher"] if args.all else (args.build_sft or [])
    for flavour in sft_flavours:
        build_datasets.build_sft(flavour=flavour, seed=args.seed)


if __name__ == "__main__":
    main()
