#!/usr/bin/env python
"""Judge-reliability check (Section 2.1): re-score sampled turns with GPT-5-mini
and report Pearson r vs the Claude-Sonnet-4 ratings and % within one point.

Usage:
    python scripts/validate_judge.py results/records/gemma-3-27b-it.jsonl [--n 260]
"""

from __future__ import annotations

import argparse

from emotional_instability.analysis.judge_agreement import validate_judge
from emotional_instability.eval.datatypes import read_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="+", help="Scored JSONL record files.")
    ap.add_argument("--n", type=int, default=260)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    records = [r for path in args.records for r in read_records(path)]
    res = validate_judge(records, n_sample=args.n, seed=args.seed)
    print(f"n               = {res.n}")
    print(f"Pearson r       = {res.pearson_r:.3f}")
    print(f"p-value         = {res.p_value:.3g}")
    print(f"% within 1 point= {res.pct_within_one:.1f}%")


if __name__ == "__main__":
    main()
