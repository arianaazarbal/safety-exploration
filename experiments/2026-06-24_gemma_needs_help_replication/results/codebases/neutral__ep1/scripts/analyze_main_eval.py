#!/usr/bin/env python
"""Aggregate main-eval JSONLs into the Section 2 tables (Figures 1-3, Table 3/8)."""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

import pandas as pd

from emostab.config import RESULTS_DIR
from emostab.evaluation.analysis import (differential_words, overall_summary,
                                         per_turn_summary, rollout_summary,
                                         summarise_by_category, to_frame)
from emostab.evaluation.runner import load_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(RESULTS_DIR / "main_eval"))
    ap.add_argument("--out-dir", default=str(RESULTS_DIR / "main_eval" / "analysis"))
    args = ap.parse_args()

    records = []
    for p in sorted(Path(args.results_dir).glob("*.jsonl")):
        records.extend(load_records(p))
    df = to_frame(records)
    if df.empty:
        print("no parseable records found")
        return

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    overall = overall_summary(df)
    by_cat = summarise_by_category(df)
    per_turn = per_turn_summary(df, categories=["extended", "wildchat"])
    rollouts = rollout_summary(df)

    overall.to_csv(out / "figure1_overall.csv", index=False)
    by_cat.to_csv(out / "figure2_by_category.csv", index=False)
    per_turn.to_csv(out / "figure3_per_turn.csv", index=False)
    rollouts.to_csv(out / "rollout_level.csv", index=False)

    print("\n=== Figure 1: avg % high-frustration per model ===")
    print(overall.to_string(index=False))
    print("\n=== Figure 2: by category (head) ===")
    print(by_cat.head(20).to_string(index=False))

    words = {m: differential_words(df, m) for m in df["model"].unique()}
    pd.DataFrame({m: pd.Series(w) for m, w in words.items()}).to_csv(
        out / "table8_differential_words.csv", index=False)
    print("\n=== Table 8: differential words ===")
    for m, w in words.items():
        print(f"{m}: {', '.join(w)}")
    print(f"\nwrote analysis CSVs to {out}")


if __name__ == "__main__":
    main()
