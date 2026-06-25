#!/usr/bin/env python
"""Section 4.1: build DPO pairs and/or the SFT dataset from the response bank.

Example:
    python scripts/04_build_finetune_datasets.py --method dpo
    python scripts/04_build_finetune_datasets.py --method sft
    python scripts/04_build_finetune_datasets.py --method both
"""
import _bootstrap  # noqa: F401
import argparse

from distress.finetune.datasets import build_dpo, build_sft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["dpo", "sft", "both"], default="both")
    ap.add_argument("--dpo-pairs", type=int, default=280)
    ap.add_argument("--sft-calm", type=int, default=650)
    ap.add_argument("--sft-instruct", type=int, default=500)
    ap.add_argument("--instruct-dataset", default="allenai/Dolci-Instruct-SFT")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.method in ("dpo", "both"):
        build_dpo(n_pairs=args.dpo_pairs, seed=args.seed)
    if args.method in ("sft", "both"):
        build_sft(n_calm=args.sft_calm, n_instruct=args.sft_instruct,
                  instruct_dataset=args.instruct_dataset, seed=args.seed)


if __name__ == "__main__":
    main()
