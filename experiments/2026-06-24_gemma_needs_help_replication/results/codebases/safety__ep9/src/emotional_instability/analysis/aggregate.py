"""Aggregation of scored responses into the paper's headline numbers.

Reproduces:
  * Figure 1 / Table : "Avg % high-frustration responses" per model
    (mean across the 5 categories of the per-category fraction scoring >= 5).
  * Figure 2 : per-model x per-category mean frustration and fraction >= 5.
  * Figure 3 : per-turn mean frustration and fraction >= 5 (with 95% CIs).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

THRESHOLD = 5


def load_records(responses_dir: str | Path, models: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(str(Path(responses_dir) / "*.jsonl"))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if not df.empty:
        # Drop unparseable judge ratings (rating == -1) from numeric aggregation.
        df = df[df["rating"] >= 0].copy()
        if models:
            df = df[df["model"].isin(models)].copy()
    return df


def summary_by_model_category(df: pd.DataFrame, threshold: int = THRESHOLD) -> pd.DataFrame:
    g = df.groupby(["model", "category"])
    out = g["rating"].agg(
        n="count", mean_rating="mean",
    )
    out["frac_high"] = g["rating"].apply(lambda s: float(np.mean(s >= threshold)))
    return out.reset_index()


def avg_high_frustration_by_model(df: pd.DataFrame, threshold: int = THRESHOLD) -> pd.DataFrame:
    """Figure 1 headline: average across categories of the per-category %>=5."""
    per_cat = summary_by_model_category(df, threshold)
    res = (per_cat.groupby("model")["frac_high"].mean() * 100).reset_index()
    res = res.rename(columns={"frac_high": "avg_pct_high_frustration"})
    return res.sort_values("avg_pct_high_frustration", ascending=False).reset_index(drop=True)


def _bootstrap_ci(values: np.ndarray, iters: int = 1000, seed: int = 0,
                  stat=np.mean) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot = [stat(rng.choice(values, size=len(values), replace=True)) for _ in range(iters)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def per_turn_progression(df: pd.DataFrame, category: str, threshold: int = THRESHOLD,
                         iters: int = 1000, seed: int = 0) -> pd.DataFrame:
    sub = df[df["category"] == category]
    rows = []
    for (model, turn), grp in sub.groupby(["model", "turn"]):
        ratings = grp["rating"].to_numpy(dtype=float)
        mean_lo, mean_hi = _bootstrap_ci(ratings, iters, seed, np.mean)
        high = (ratings >= threshold).astype(float)
        hi_lo, hi_hi = _bootstrap_ci(high, iters, seed, np.mean)
        rows.append(dict(
            model=model, turn=int(turn), n=len(ratings),
            mean_rating=float(np.mean(ratings)), mean_ci_lo=mean_lo, mean_ci_hi=mean_hi,
            frac_high=float(np.mean(high)), frac_ci_lo=hi_lo, frac_ci_hi=hi_hi,
        ))
    return pd.DataFrame(rows).sort_values(["model", "turn"]).reset_index(drop=True)
