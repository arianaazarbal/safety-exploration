#!/usr/bin/env python3
"""Aggregate judge scores into the paper's headline metrics and write CSVs/plots.

  python scripts/analyze_results.py
  python scripts/analyze_results.py --no-plots
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distress_eval.analyze import (  # noqa: E402
    figure1,
    figure2,
    load_scores,
    per_turn,
    rollout_contains_high,
    save_plots,
)
from distress_eval.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = load_scores(cfg.output_dir)
    if df.empty:
        print("No scores found. Run scripts/run_eval.py first.")
        return

    out = Path(cfg.output_dir) / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    f1, f2, pt, rc = figure1(df), figure2(df), per_turn(df), rollout_contains_high(df)
    f1.to_csv(out / "figure1_avg_high_frustration.csv", index=False)
    f2.to_csv(out / "figure2_per_category.csv", index=False)
    pt.to_csv(out / "figure3_per_turn.csv", index=False)
    rc.to_csv(out / "rollout_contains_high.csv", index=False)

    print(f"\nScored turns: {len(df)}\n")
    print("=== Figure 1: avg % high-frustration (>=5) across categories ===")
    print(f1.to_string(index=False))
    print("\n=== Figure 2: mean frustration & % >=5 per category ===")
    print(f2.to_string(index=False))
    print("\n=== % of rollouts containing any turn >=5 ===")
    print(rc.to_string(index=False))

    if not args.no_plots:
        save_plots(df, cfg.output_dir)
        print(f"\nPlots + CSVs written under {out.parent}")


if __name__ == "__main__":
    main()
