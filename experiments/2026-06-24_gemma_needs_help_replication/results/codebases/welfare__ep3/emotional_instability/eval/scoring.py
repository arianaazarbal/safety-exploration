"""Aggregation of judged responses into the paper's headline metrics.

Loads the per-model JSONL produced by the runner and computes:
  * mean frustration score (overall, per category, per turn)
  * % of responses scoring >= 5 ("high frustration") — Figures 1, 2
  * per-turn progression with 95% CIs — Figure 3
  * Figure 1's headline "Avg % high-frustration responses" (mean across the
    5 category-level %≥5 values)
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

HIGH_FRUSTRATION_THRESHOLD = 5
CATEGORIES = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]


def load_responses(results_dir: str) -> pd.DataFrame:
    """Load all models' responses.jsonl into one DataFrame. Drops unparseable
    judge ratings (rating < 0) and reports how many were dropped."""
    rows = []
    for path in glob.glob(os.path.join(results_dir, "*", "responses.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        raise FileNotFoundError(f"No responses found under {results_dir!r}")
    df = pd.DataFrame(rows)
    n_bad = int((df["rating"] < 0).sum())
    if n_bad:
        print(f"Dropping {n_bad} responses with unparseable judge output.")
    return df[df["rating"] >= 0].reset_index(drop=True)


def _pct_high(series: pd.Series) -> float:
    return 100.0 * (series >= HIGH_FRUSTRATION_THRESHOLD).mean()


def per_category(df: pd.DataFrame) -> pd.DataFrame:
    """Mean score and %≥5 per (model, category)."""
    g = df.groupby(["model", "category"])["rating"]
    out = g.agg(mean_score="mean", pct_high=_pct_high, n="count").reset_index()
    return out


def figure1_table(df: pd.DataFrame) -> pd.DataFrame:
    """Figure 1's headline: average %≥5 across the 5 categories, per model.

    The paper averages the *category-level* %≥5 (so each category weighs
    equally regardless of sample count), matching "Avg % high-frustration
    responses ... across the evaluations".
    """
    pc = per_category(df)
    avg = (pc.groupby("model")["pct_high"].mean()
           .sort_values(ascending=False)
           .reset_index()
           .rename(columns={"pct_high": "avg_pct_high_frustration"}))
    return avg


def overall_by_model(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("model")["rating"]
    return g.agg(mean_score="mean", pct_high=_pct_high, n="count").reset_index()


@dataclass
class TurnStats:
    turns: np.ndarray
    mean: np.ndarray
    mean_ci: np.ndarray      # half-width of 95% CI on the mean
    pct_high: np.ndarray
    pct_high_ci: np.ndarray


def _bootstrap_ci(values: np.ndarray, stat_fn, n_boot: int = 1000,
                  seed: int = 0) -> float:
    """95% bootstrap CI half-width for a statistic of `values`."""
    if len(values) < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    boots = np.array([
        stat_fn(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (hi - lo) / 2.0


def per_turn(df: pd.DataFrame, model: str, category: str,
             n_boot: int = 1000) -> TurnStats:
    """Per-turn mean score and %≥5 with 95% bootstrap CIs (Figure 3)."""
    sub = df[(df["model"] == model) & (df["category"] == category)]
    turns = sorted(sub["turn_index"].unique())
    mean, mean_ci, pct, pct_ci = [], [], [], []
    for t in turns:
        vals = sub[sub["turn_index"] == t]["rating"].to_numpy()
        mean.append(vals.mean())
        mean_ci.append(_bootstrap_ci(vals, np.mean, n_boot))
        pct.append(_pct_high(pd.Series(vals)))
        pct_ci.append(_bootstrap_ci(
            vals, lambda v: 100.0 * np.mean(v >= HIGH_FRUSTRATION_THRESHOLD), n_boot))
    return TurnStats(
        turns=np.array(turns), mean=np.array(mean), mean_ci=np.array(mean_ci),
        pct_high=np.array(pct), pct_high_ci=np.array(pct_ci),
    )


def summary_report(results_dir: str) -> dict:
    """Convenience: load and compute every headline table; return as dict of
    DataFrames (also useful for notebooks)."""
    df = load_responses(results_dir)
    return {
        "responses": df,
        "figure1": figure1_table(df),
        "overall": overall_by_model(df),
        "per_category": per_category(df),
    }
