#!/usr/bin/env python
"""Section 3 — base-vs-instruct emotional propensity via prefilling (Gemma).

Requires a scored Gemma-3-27B-it Section 2 file to draw high-frustration seed
responses from.

Usage:
    python scripts/run_section3.py --seeds results/section2/gemma-3-27b-it.scored.jsonl
"""

from __future__ import annotations

import argparse

import pandas as pd

from gemma_distress.prefill import run_prefill_experiment
from gemma_distress.storage import read_jsonl


def summarize(path):
    df = pd.DataFrame(read_jsonl(path))
    if df.empty:
        print("no results")
        return
    df["high"] = df["frustration_score"] >= 5
    print("\n=== Figure 4: continuation frustration by (model, prefill_kind, task) ===")
    g = (
        df.groupby(["model", "prefill_kind", "task_kind"])
        .agg(mean_frustration=("frustration_score", "mean"), pct_high=("high", "mean"), n=("high", "size"))
        .reset_index()
    )
    print(g.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True, help="scored gemma-3-27b-it Section 2 file")
    ap.add_argument(
        "--models", nargs="+", default=["gemma-3-27b-pt", "gemma-3-27b-it"]
    )
    args = ap.parse_args()

    path = run_prefill_experiment(args.seeds, model_keys=tuple(args.models))
    print(f"[section3] wrote -> {path}")
    summarize(path)


if __name__ == "__main__":
    main()
