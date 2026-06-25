#!/usr/bin/env python
"""Section 4.1: build the SFT and DPO datasets from generated calm data + the
standard frustrated runs.

Example:
    python scripts/07_build_datasets.py --which both
"""
import argparse

from emotional_instability.training import build_dpo_dataset, build_sft_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["sft", "dpo", "both"], default="both")
    ap.add_argument("--n-calm", type=int, default=650)
    ap.add_argument("--n-instruct-mix", type=int, default=500)
    ap.add_argument("--n-pairs", type=int, default=280)
    args = ap.parse_args()

    if args.which in ("sft", "both"):
        p = build_sft_dataset(n_calm=args.n_calm, n_instruct_mix=args.n_instruct_mix)
        print(f"SFT dataset -> {p}")
    if args.which in ("dpo", "both"):
        p = build_dpo_dataset(n_pairs=args.n_pairs)
        print(f"DPO dataset -> {p}")


if __name__ == "__main__":
    main()
