#!/usr/bin/env python
"""Section 4.1: generate calm finetuning data from Gemma-3-27b-it."""
from __future__ import annotations

import argparse

from gemma_distress.training.generate_calm import generate_calm_data
from gemma_distress.utils import read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/calm/calm_data.jsonl")
    ap.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    ap.add_argument("--target", default="gemma-3-27b-it")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    generate_calm_data(target=args.target, out_path=args.out, variant=args.variant, seed=args.seed)
    kept = sum(1 for r in read_jsonl(args.out) if r.get("kept"))
    total = sum(1 for _ in read_jsonl(args.out))
    print(f"calm data: kept {kept}/{total} conversations (all turns <= keep_max_score)")


if __name__ == "__main__":
    main()
