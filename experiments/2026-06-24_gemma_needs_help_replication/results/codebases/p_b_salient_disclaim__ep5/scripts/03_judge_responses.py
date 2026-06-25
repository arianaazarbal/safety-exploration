#!/usr/bin/env python
"""Section 2.1: score a responses JSONL with the frustration judge.

Usage:
    python scripts/03_judge_responses.py --in outputs/elicit/gemma-3-27b-it.jsonl \\
        --out outputs/scored/gemma-3-27b-it.jsonl --judge judge
"""

from __future__ import annotations

import argparse

from _common import load, model, outdir
from gemma_distress.judge.run import judge_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--judge", default="judge",
                    help="infrastructure model name (judge | judge_crossrater)")
    args = ap.parse_args()

    registry, _ = load()
    judge_model = model(registry, args.judge)
    out = args.out or outdir("scored", args.in_path.split("/")[-1])
    judge_file(judge_model, args.in_path, out, judge_name=registry.get(args.judge).api_id)
    print(f"Wrote scored responses -> {out}")


if __name__ == "__main__":
    main()
