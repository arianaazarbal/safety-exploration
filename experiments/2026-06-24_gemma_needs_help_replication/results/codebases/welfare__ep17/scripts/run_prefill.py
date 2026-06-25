#!/usr/bin/env python3
"""Section 3: base-vs-instruct prefill experiment (Gemma).

Collects high-frustration seed conversations from Gemma-27B-it, builds
early/onset truncations, paraphrases them, generates continuations from each
base/instruct model, judges them, and prints the divergence summary (Figure 4).
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from emotional_instability.config import load_config
from emotional_instability.prefill.continuation import run_continuations


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    records = run_continuations(cfg)
    rows = [r.__dict__ for r in records]
    df = pd.DataFrame(rows)
    if df.empty:
        print("no continuations produced (no high-frustration seeds?)")
        return

    threshold = int(cfg["evaluation"]["high_frustration_threshold"])
    df["high"] = df["score"] >= threshold
    summary = (df.groupby(["category", "truncation", "model_kind", "model"])
                 .agg(mean_score=("score", "mean"),
                      pct_high=("high", lambda s: 100 * s.mean()),
                      n=("score", "size"))
                 .round(2).reset_index())
    out = cfg.path_for("scores") / "prefill_summary.csv"
    summary.to_csv(out, index=False)
    print(summary.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
