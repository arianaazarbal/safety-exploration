#!/usr/bin/env python3
"""Judge-reliability cross-check (Section 2.1).

Re-scores a random subset of an existing responses.jsonl with a secondary judge
(GPT-5-mini via OpenRouter) and reports Pearson r and % within one point of the
primary Claude-Sonnet-4 ratings.

    python validate_judge.py --output-dir ./outputs --n 260
"""

from __future__ import annotations

import argparse
import json
import os

from distress_eval.analyze import load_records
from distress_eval.judge_validation import cross_check


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default="./outputs")
    ap.add_argument("--n", type=int, default=260)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    records = load_records(os.path.join(args.output_dir, "responses.jsonl"))
    stats = cross_check(records, n=args.n, seed=args.seed)
    print(json.dumps(stats, indent=2))
    with open(os.path.join(args.output_dir, "judge_validation.json"), "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
