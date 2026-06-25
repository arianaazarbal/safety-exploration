#!/usr/bin/env python
"""Reproduce the inter-judge agreement check (Pearson r) on an elicitation file."""
from __future__ import annotations

import argparse
import json

from gemma_distress.eval.judge_agreement import validate_judge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="elicitation output file")
    ap.add_argument("--n", type=int, default=260)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(json.dumps(validate_judge(args.jsonl, n_sample=args.n, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
