#!/usr/bin/env python
"""Build SFT and DPO datasets from generated calm/frustrated data (Section 4.1).

Example
-------
python scripts/build_datasets.py \
    --calm outputs/data/calm.jsonl \
    --frustrated outputs/eval/gemma-3-27b-it.jsonl \
    --out-dir outputs/data
"""
from __future__ import annotations

import argparse
import os

import _common  # noqa: F401

from instability.training import build_dpo_dataset, build_sft_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calm", required=True, help="calm conversations JSONL")
    ap.add_argument("--frustrated", required=True,
                    help="frustrated source JSONL (calm-gen or main-eval numeric)")
    ap.add_argument("--out-dir", default="outputs/data")
    ap.add_argument("--target-pairs", type=int, default=280)
    ap.add_argument("--no-hf-mix", action="store_true",
                    help="skip Dolci-Instruct-SFT mix-in for SFT")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    build_sft_dataset(
        args.calm, os.path.join(args.out_dir, "sft.jsonl"),
        use_hf_mix=not args.no_hf_mix,
    )
    build_dpo_dataset(
        args.calm, args.frustrated, os.path.join(args.out_dir, "dpo.jsonl"),
        target_pairs=args.target_pairs,
    )


if __name__ == "__main__":
    main()
