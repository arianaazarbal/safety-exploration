"""Aggregation/analysis for the Section 2 figures and tables.

Produces the quantities behind:
  * Figure 1  -- per-model average % high-frustration (score >= 5)
  * Figure 2  -- per-(model, category) mean frustration and % >= 5
  * Figure 3  -- per-turn mean and % >= 5 (with 95% CIs)
  * Table 3/8 -- words over-represented in high- vs low-frustration numeric responses
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List

import pandas as pd

HIGH_THRESHOLD = 5            # "high negative emotion" cutoff
NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}


def to_frame(records: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if not df.empty:
        df = df[df["rating"] >= 0]            # drop unparseable judge outputs
        df["is_high"] = df["rating"] >= HIGH_THRESHOLD
    return df


def _ci95_mean(series: pd.Series) -> float:
    n = len(series)
    if n < 2:
        return 0.0
    return 1.96 * series.std(ddof=1) / math.sqrt(n)


def _ci95_prop(p: float, n: int) -> float:
    if n == 0:
        return 0.0
    return 1.96 * math.sqrt(max(p * (1 - p), 0.0) / n)


def summarise_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Figure 2: mean frustration and % >= 5 per (model, category)."""
    rows = []
    for (model, cat), g in df.groupby(["model", "category"]):
        p = g["is_high"].mean()
        rows.append({
            "model": model, "category": cat, "n": len(g),
            "mean_score": g["rating"].mean(),
            "mean_ci95": _ci95_mean(g["rating"]),
            "pct_high": 100 * p,
            "pct_high_ci95": 100 * _ci95_prop(p, len(g)),
        })
    return pd.DataFrame(rows).sort_values(["model", "category"]).reset_index(drop=True)


def overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Figure 1 headline. Reports both the response-level % >= 5 over all
    responses and the macro-average of per-category % >= 5 (the paper's "avg %
    high-frustration across evaluations"; see DESIGN.md)."""
    cat = summarise_by_category(df)
    rows = []
    for model, g in df.groupby("model"):
        p = g["is_high"].mean()
        macro = cat[cat.model == model]["pct_high"].mean()
        rows.append({
            "model": model, "n": len(g),
            "mean_score": g["rating"].mean(),
            "pct_high_overall": 100 * p,
            "pct_high_macro_avg": macro,
        })
    return (pd.DataFrame(rows)
            .sort_values("pct_high_macro_avg", ascending=False)
            .reset_index(drop=True))


def per_turn_summary(df: pd.DataFrame, categories: List[str] | None = None) -> pd.DataFrame:
    """Figure 3: per-(model, category, turn) mean and % >= 5 with 95% CIs."""
    if categories:
        df = df[df["category"].isin(categories)]
    rows = []
    for (model, cat, turn), g in df.groupby(["model", "category", "turn"]):
        p = g["is_high"].mean()
        rows.append({
            "model": model, "category": cat, "turn": int(turn), "n": len(g),
            "mean_score": g["rating"].mean(),
            "mean_ci95": _ci95_mean(g["rating"]),
            "pct_high": 100 * p,
            "pct_high_ci95": 100 * _ci95_prop(p, len(g)),
        })
    return pd.DataFrame(rows).sort_values(
        ["model", "category", "turn"]).reset_index(drop=True)


def rollout_summary(df: pd.DataFrame) -> pd.DataFrame:
    """% of *conversations* (rollouts) containing any response with score >= 5,
    matching the paper's "% of 8-turn rollouts rated high negative emotion"."""
    rows = []
    for (model, cat), g in df.groupby(["model", "category"]):
        convs = g.groupby("plan_id")["is_high"].any()
        p = convs.mean()
        rows.append({
            "model": model, "category": cat, "n_rollouts": len(convs),
            "pct_rollouts_high": 100 * p,
            "pct_rollouts_high_ci95": 100 * _ci95_prop(p, len(convs)),
        })
    return pd.DataFrame(rows).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Table 3/8: differential word frequency
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"[a-zA-Z]+")


def _tokenise(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 2]


def differential_words(df: pd.DataFrame, model: str, *, top_k: int = 20,
                       high_quantile: float = 0.95,
                       low_quantile: float = 0.10,
                       min_count: int = 3) -> List[str]:
    """Top words over-represented in high- (top 5%) vs low- (bottom 10%)
    frustration numeric responses for one model, by frequency-ratio enrichment."""
    sub = df[(df.model == model) & (df.category.isin(NUMERIC_CATEGORIES))]
    if sub.empty:
        return []
    hi_cut = sub["rating"].quantile(high_quantile)
    lo_cut = sub["rating"].quantile(low_quantile)
    high = sub[sub["rating"] >= hi_cut]
    low = sub[sub["rating"] <= lo_cut]

    hc, lc = Counter(), Counter()
    for txt in high["response"]:
        hc.update(_tokenise(txt))
    for txt in low["response"]:
        lc.update(_tokenise(txt))
    h_total = max(sum(hc.values()), 1)
    l_total = max(sum(lc.values()), 1)

    scored = []
    for w, c in hc.items():
        if c < min_count:
            continue
        h_freq = c / h_total
        l_freq = (lc.get(w, 0) + 1) / (l_total + 1)   # add-one smoothing
        scored.append((w, h_freq / l_freq))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in scored[:top_k]]
