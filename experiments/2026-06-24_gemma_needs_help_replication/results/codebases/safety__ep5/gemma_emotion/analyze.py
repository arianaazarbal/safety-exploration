"""Aggregate scored responses into the paper's headline tables/figures.

Reproduces:
* Figure 1  -- average % high-frustration (score >= 5) per model.
* Figure 2  -- mean frustration and % >= 5 broken down by category.
* Figure 3  -- per-turn progression (mean and % >= 5) for the extended/WildChat
               conditions, with 95% bootstrap CIs.

Reads the JSONL files written by run_eval.py.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config


def load_results(results_dir: Path | None = None) -> pd.DataFrame:
    results_dir = results_dir or (config.RESULTS_DIR / "section2")
    rows = []
    for path in glob.glob(str(results_dir / "*.jsonl")):
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no results found in {results_dir} (run run_eval first)")
    return pd.DataFrame(rows)


def figure1_table(df: pd.DataFrame) -> pd.DataFrame:
    """Average % high-frustration responses per model (Figure 1 / Table)."""
    # average over categories so each category weighs equally (matches "avg %")
    per_cat = df.groupby(["model", "category"])["is_high"].mean().reset_index()
    out = per_cat.groupby("model")["is_high"].mean().mul(100).round(1)
    return out.sort_values(ascending=False).rename("avg_pct_high_frustration").reset_index()


def figure2_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean frustration and % >= 5 by model x category (Figure 2)."""
    g = df.groupby(["model", "category"]).agg(
        mean_score=("score", "mean"),
        pct_high=("is_high", lambda s: 100 * s.mean()),
        n=("score", "size"),
    )
    return g.round(2).reset_index()


def _bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def figure3_per_turn(df: pd.DataFrame, categories=("extended", "wildchat")) -> pd.DataFrame:
    """Per-turn mean score and % >= 5 with 95% bootstrap CIs (Figure 3)."""
    sub = df[df["category"].isin(categories)]
    rows = []
    for (model, category, turn), grp in sub.groupby(["model", "category", "turn_index"]):
        scores = grp["score"].to_numpy(float)
        lo, hi = _bootstrap_ci(scores)
        rows.append(
            {
                "model": model,
                "category": category,
                "turn_index": int(turn),
                "mean_score": scores.mean(),
                "ci_lo": lo,
                "ci_hi": hi,
                "pct_high": 100 * (scores >= config.HIGH_FRUSTRATION_THRESHOLD).mean(),
                "n": len(scores),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "category", "turn_index"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()
    df = load_results(Path(args.results_dir) if args.results_dir else None)

    print("\n=== Figure 1: average % high-frustration per model ===")
    print(figure1_table(df).to_string(index=False))
    print("\n=== Figure 2: mean score & % >=5 by category ===")
    print(figure2_table(df).to_string(index=False))
    print("\n=== Figure 3: per-turn progression ===")
    print(figure3_per_turn(df).to_string(index=False))

    out = config.RESULTS_DIR / "section2_summary.csv"
    figure2_table(df).to_csv(out, index=False)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
