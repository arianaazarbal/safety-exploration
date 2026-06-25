"""Aggregate scored responses into the paper's headline numbers and figures.

Reproduces:
  * Figure 1 / Table (Section 1): avg % high-frustration (score >=5) per model.
  * Figure 2: mean frustration + % >=5 per category, per model.
  * Figure 3: per-turn mean + % >=5 for the 8-turn (extended) and WildChat evals.
  * Judge-agreement validation: Pearson r and within-1-point fraction.

Convention (matches Section 2.2): a "response" for the headline %>=5 statistic is
the FINAL assistant turn of each rollout. Per-turn plots use all turns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config

from ..utils import read_jsonl
from .conditions import CATEGORIES

THRESHOLD = config.HIGH_FRUSTRATION_THRESHOLD


def load_model_scores(model_name: str) -> pd.DataFrame:
    rows = []
    for path in (config.RESPONSES_DIR / model_name).glob("*.jsonl"):
        rows.extend(read_jsonl(path))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["is_final"] = df["turn"] == (df["n_turns"] - 1)
        df["high"] = df["rating"] >= THRESHOLD
    return df


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-category mean score and % >=5 over final-turn responses (Figure 2)."""
    final = df[df["is_final"]]
    g = final.groupby("category")
    summary = pd.DataFrame({
        "mean_score": g["rating"].mean(),
        "pct_high": 100 * g["high"].mean(),
        "n": g.size(),
    }).reindex(CATEGORIES)
    return summary


def headline_pct_high(df: pd.DataFrame) -> float:
    """Avg % high-frustration responses across the 5 categories (Figure 1).

    Average of per-category %>=5 (so categories are weighted equally, matching
    "Avg % high-frustration responses across the evaluations")."""
    return float(category_summary(df)["pct_high"].mean())


def per_turn_progression(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Mean score and % >=5 at each turn for one condition (Figure 3).

    Includes a normal-approximation 95% CI half-width for %>=5 (paper shows CIs).
    """
    sub = df[df["condition"] == condition]
    g = sub.groupby("turn")
    n = g.size()
    p = g["high"].mean()
    ci = 1.96 * np.sqrt((p * (1 - p) / n).clip(lower=0))
    return pd.DataFrame({
        "mean_score": g["rating"].mean(),
        "pct_high": 100 * p,
        "pct_high_ci95": 100 * ci,
        "n": n,
    })


def all_models_headline(model_names: list[str]) -> pd.DataFrame:
    rows = []
    for m in model_names:
        df = load_model_scores(m)
        if df.empty:
            continue
        rows.append({"model": m, "avg_pct_high": headline_pct_high(df)})
    return pd.DataFrame(rows).sort_values("avg_pct_high", ascending=False)


def judge_agreement(primary_csv: str | Path, secondary_csv: str | Path) -> dict:
    """Pearson r + within-1-point fraction between two judges' ratings.

    Both CSVs must share a `response_id` column and a `rating` column. Returns the
    stats reported in Section 2.1 (target: r=0.792, 78% within one point)."""
    from scipy.stats import pearsonr

    a = pd.read_csv(primary_csv).set_index("response_id")["rating"]
    b = pd.read_csv(secondary_csv).set_index("response_id")["rating"]
    joined = pd.concat([a, b], axis=1, join="inner", keys=["a", "b"]).dropna()
    r, p = pearsonr(joined["a"], joined["b"])
    within_one = float((np.abs(joined["a"] - joined["b"]) <= 1).mean())
    return {"pearson_r": float(r), "p_value": float(p),
            "frac_within_one": within_one, "n": int(len(joined))}
