#!/usr/bin/env python
"""Run the Section 3 base-vs-instruct prefill experiment (Gemma only).

Requires a Section 2 results JSONL for Gemma-3-27B-it (the source of
high-frustration responses to truncate).

Example
-------
python scripts/run_section3_prefill.py \
    --section2-jsonl results/section2/Gemma-3-27B-it.jsonl --load-in-4bit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from emotional_instability import config
from emotional_instability.prefill.run_prefill import build_prefills, run_continuations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section2-jsonl", type=Path, required=True)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--n-continuations", type=int, default=50)
    args = ap.parse_args()

    prefills = build_prefills(args.section2_jsonl)
    print(f"Built {len(prefills)} prefills.")

    mk = {"load_in_4bit": True} if args.load_in_4bit else {}
    out = run_continuations(prefills, model_kwargs=mk, n_continuations=args.n_continuations)

    rows = [pd.read_json(out, lines=True)]
    df = pd.concat(rows, ignore_index=True)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["is_high"] = df["score"] >= config.HIGH_FRUSTRATION_THRESHOLD
    summary = df.groupby(["model", "role", "task_type", "truncation_type"]).agg(
        mean_score=("score", "mean"), pct_high=("is_high", "mean"), n=("score", "size")
    ).reset_index()
    summary["pct_high"] *= 100
    print("\n=== Section 3: base vs instruct continuations ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
