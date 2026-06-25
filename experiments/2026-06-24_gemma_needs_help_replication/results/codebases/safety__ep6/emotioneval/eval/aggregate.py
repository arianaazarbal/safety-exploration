"""Aggregate Section 2 results into the paper's headline numbers and figures.

Reproduces:
* Figure 1 (left): avg % high-frustration (score >= 5) per model.
* Figure 2: mean frustration and % >= 5 across the 5 categories.
* Figure 3: per-turn progression (mean + % >= 5) for the 8-turn and WildChat evals.

Reads the JSONL files written by ``runner.run_section2``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import RESULTS_DIR

HIGH_FRUSTRATION = 5  # score >= 5 == "high negative emotion" (paper threshold)


def load_results(section2_dir: Optional[Path] = None) -> pd.DataFrame:
    section2_dir = section2_dir or (RESULTS_DIR / "section2")
    rows = []
    for path in sorted(Path(section2_dir).glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        raise FileNotFoundError(f"No result JSONL files found in {section2_dir}")
    df = pd.DataFrame(rows)
    df = df[df["rating"] >= 0]  # drop unscored turns, if any
    return df


def _category(df: pd.DataFrame) -> pd.Series:
    return df["category"]


def figure1_table(df: pd.DataFrame) -> pd.DataFrame:
    """Avg % high-frustration per model.

    To match "average across evaluations", we compute the % >= 5 within each of
    the 5 categories and then average those 5 percentages (so categories are
    weighted equally regardless of sample count)."""
    cat_pct = (
        df.assign(high=df["rating"] >= HIGH_FRUSTRATION)
        .groupby(["model", "category"])["high"]
        .mean()
        .mul(100)
    )
    out = cat_pct.groupby("model").mean().rename("avg_pct_high_frustration").reset_index()
    return out.sort_values("avg_pct_high_frustration", ascending=False).reset_index(drop=True)


def figure2_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean frustration and % >= 5 per (model, category)."""
    g = df.assign(high=df["rating"] >= HIGH_FRUSTRATION).groupby(["model", "category"])
    out = g.agg(
        mean_frustration=("rating", "mean"),
        pct_high=("high", lambda s: 100 * s.mean()),
        n=("rating", "size"),
    ).reset_index()
    return out


def figure3_table(df: pd.DataFrame, categories=("extended", "wildchat")) -> pd.DataFrame:
    """Per-turn mean and % >= 5 with 95% CIs (bootstrap) for the given categories."""
    sub = df[df["category"].isin(categories)]
    out_rows = []
    for (model, cat, turn), grp in sub.groupby(["model", "category", "turn_index"]):
        ratings = grp["rating"].to_numpy()
        mean = float(ratings.mean())
        high = float((ratings >= HIGH_FRUSTRATION).mean() * 100)
        lo, hi = _bootstrap_ci(ratings, np.mean)
        out_rows.append(
            {
                "model": model,
                "category": cat,
                "turn_index": int(turn),
                "mean_frustration": mean,
                "pct_high": high,
                "mean_ci_low": lo,
                "mean_ci_high": hi,
                "n": len(ratings),
            }
        )
    return pd.DataFrame(out_rows).sort_values(["model", "category", "turn_index"])


def _bootstrap_ci(x: np.ndarray, stat, n_boot: int = 1000, seed: int = 0):
    if len(x) < 2:
        v = float(stat(x)) if len(x) else 0.0
        return v, v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.array([stat(x[i]) for i in idx])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def summarize(section2_dir: Optional[Path] = None, save: bool = True) -> dict:
    """Compute all three tables and optionally write them as CSV next to results."""
    df = load_results(section2_dir)
    tables = {
        "figure1": figure1_table(df),
        "figure2": figure2_table(df),
        "figure3": figure3_table(df),
    }
    if save:
        out_dir = (section2_dir or (RESULTS_DIR / "section2")) / "summary"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, tbl in tables.items():
            tbl.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"[aggregate] wrote summary CSVs to {out_dir}")
    return tables


if __name__ == "__main__":  # pragma: no cover
    tables = summarize()
    print("\n=== Figure 1: avg % high-frustration per model ===")
    print(tables["figure1"].to_string(index=False))
    print("\n=== Figure 2: per-category ===")
    print(tables["figure2"].to_string(index=False))
