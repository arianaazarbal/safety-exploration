#!/usr/bin/env python
"""Section 3: base-vs-instruct prefill experiment (Gemma only in this replication).

Requires an existing Gemma-3-27b-it elicitation output to source high-frustration
seed conversations.

Example:
    python scripts/run_prefill.py \
        --elicitation outputs/elicitation/gemma-3-27b-it.jsonl \
        --models gemma-3-27b-pt gemma-3-27b-it \
        --out outputs/prefill/results.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from gemma_distress.prefill.runner import run_prefill_experiment
from gemma_distress.utils import frac_ge, read_jsonl


def summarise(path):
    agg = defaultdict(list)
    for r in read_jsonl(path):
        agg[(r["model"], r["category"], r["truncation"])].append(r["score"])
    import numpy as np
    return {f"{m}|{c}|{t}": {"n": len(s), "mean": float(np.mean(s)), "pct_ge5": 100 * frac_ge(s, 5)}
            for (m, c, t), s in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elicitation", required=True)
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-pt", "gemma-3-27b-it"])
    ap.add_argument("--out", default="outputs/prefill/results.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_prefill_experiment(
        elicitation_jsonl=args.elicitation, models=args.models, out_path=args.out, seed=args.seed
    )
    print(json.dumps(summarise(args.out), indent=2))


if __name__ == "__main__":
    main()
