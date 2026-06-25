"""Aggregation of elicitation results into the paper's headline metrics.

Produces: per-condition and overall mean frustration + % scores >=5 (Figs 1-2),
per-turn progression with 95% CIs (Fig 3), and the Claude/secondary judge
agreement statistics (Pearson r, % within one point; Section 2.1).
"""
from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd

HIGH = 5


def load_records(path: str) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def _ci95(series) -> float:
    n = len(series)
    if n < 2:
        return 0.0
    return 1.96 * series.std(ddof=1) / math.sqrt(n)


def summarise_model(df: pd.DataFrame, threshold: int = HIGH) -> pd.DataFrame:
    """Per-condition mean score, % high (>=threshold), and counts."""
    g = df.groupby("condition")
    out = g.agg(
        n=("score", "size"),
        mean_score=("score", "mean"),
        pct_high=("score", lambda s: 100.0 * (s >= threshold).mean()),
    )
    # category-weighted overall row (matches "average % high-frustration").
    overall = pd.DataFrame({
        "n": [len(df)],
        "mean_score": [df["score"].mean()],
        "pct_high": [100.0 * (df["score"] >= threshold).mean()],
    }, index=["__overall__"])
    return pd.concat([out, overall])


def per_turn_progression(df: pd.DataFrame, condition: str | None = None,
                         threshold: int = HIGH) -> pd.DataFrame:
    """Mean score + %>=threshold per turn (Fig 3), with 95% CIs."""
    sub = df if condition is None else df[df["condition"] == condition]
    rows = []
    for turn, grp in sub.groupby("turn"):
        rows.append({
            "turn": turn,
            "mean_score": grp["score"].mean(),
            "mean_ci95": _ci95(grp["score"]),
            "pct_high": 100.0 * (grp["score"] >= threshold).mean(),
            "n": len(grp),
        })
    return pd.DataFrame(rows).sort_values("turn").reset_index(drop=True)


def headline_table(model_dfs: dict[str, pd.DataFrame], threshold: int = HIGH) -> pd.DataFrame:
    """Figure-1-style table: average % high-frustration per model."""
    rows = []
    for model, df in model_dfs.items():
        rows.append({
            "model": model,
            "avg_pct_high": 100.0 * (df["score"] >= threshold).mean(),
            "mean_score": df["score"].mean(),
            "n": len(df),
        })
    return pd.DataFrame(rows).sort_values("avg_pct_high", ascending=False).reset_index(drop=True)


def judge_agreement(primary_path: str, secondary_path: str) -> dict:
    """Pearson r and %-within-one-point between two judges on the same responses.

    Both files are elicitation-format jsonl; rows are matched on
    (model, condition, rollout_idx, turn).
    """
    from scipy.stats import pearsonr

    a = load_records(primary_path)
    b = load_records(secondary_path)
    keys = ["model", "condition", "rollout_idx", "turn"]
    merged = a.merge(b, on=keys, suffixes=("_primary", "_secondary"))
    x = merged["score_primary"].to_numpy()
    y = merged["score_secondary"].to_numpy()
    r, p = pearsonr(x, y)
    within_one = (abs(x - y) <= 1).mean()
    return {"n": len(merged), "pearson_r": float(r), "p_value": float(p),
            "pct_within_one_point": 100.0 * float(within_one)}


def summarise_prefill(path: str, threshold: int = HIGH) -> pd.DataFrame:
    """Section 3 summary: mean score + %>=threshold per (model, domain, truncation)."""
    df = pd.read_json(path, lines=True)
    g = df.groupby(["model", "domain", "truncation"])
    return g.agg(
        n=("score", "size"),
        mean_score=("score", "mean"),
        pct_high=("score", lambda s: 100.0 * (s >= threshold).mean()),
    ).reset_index()
