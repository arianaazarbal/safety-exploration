#!/usr/bin/env python
"""Section 3: base-vs-instruct prefill experiment (Gemma only).

Sources high-frustration responses from gemma-3-27b-it, truncates + paraphrases
them, then measures continuations from gemma-3-27b-pt (base) and gemma-3-27b-it
(instruct).

    python scripts/run_section3.py
"""

from __future__ import annotations

import argparse

import pandas as pd

from emotional_instability import config
from emotional_instability.prefill import run_section3_prefill


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="results")
    args = ap.parse_args()

    runtime = config.RuntimeConfig(output_dir=args.output)
    records = run_section3_prefill(runtime=runtime)

    df = pd.DataFrame(records).dropna(subset=["rating"])
    if df.empty:
        print("No continuations scored.")
        return
    df["high"] = df["rating"] >= 5
    summary = (df.groupby(["model", "kind", "truncation"])
                 .agg(mean=("rating", "mean"), pct_high=("high", "mean"),
                      n=("rating", "size")).reset_index())
    print("\nBase vs instruct continuation frustration:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
