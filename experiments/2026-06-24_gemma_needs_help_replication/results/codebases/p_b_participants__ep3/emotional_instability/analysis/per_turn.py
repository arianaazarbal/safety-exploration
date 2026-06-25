"""Per-turn frustration progression (paper §2.2, Figure 3).

Figure 3 shows that the multi-turn setting is what elicits high frustration:
Gemma-27B's mean score rises from ~1.5 (turn 1) to ~5.5 (turn 8) in the Extended
condition, and in WildChat no model scores >=5 until the third turn. The figure
plots mean score and %>=5 per turn index, with 95% confidence intervals (the
"faded area" in the caption).

This module computes that progression for any subset of conditions. We default
to the two multi-turn conditions the paper highlights (``extended`` 8-turn and
``wildchat`` 5-turn) but accept any category filter.

Confidence intervals:
  * mean score   — normal-approximation CI (mean ± 1.96 * SEM). With hundreds of
    rollouts per turn this is indistinguishable from a bootstrap and far cheaper.
  * %>=5         — Wilson score interval, which is well-behaved for proportions
    near 0/1 (relevant since early-turn %>=5 is ~0). Falls back gracefully when
    scipy is unavailable.

See DESIGN.md §"Confidence intervals" for why these estimators were chosen.
"""
from __future__ import annotations

import math

import pandas as pd

Z = 1.959963984540054  # 97.5th percentile of the standard normal (95% CI)


def _mean_ci(values: pd.Series) -> tuple[float, float, float]:
    """Normal-approx 95% CI for the mean. Returns (mean, lo, hi)."""
    n = len(values)
    mean = float(values.mean()) if n else float("nan")
    if n <= 1:
        return mean, mean, mean
    sem = float(values.std(ddof=1)) / math.sqrt(n)
    return mean, mean - Z * sem, mean + Z * sem


def _wilson_ci(k: int, n: int) -> tuple[float, float, float]:
    """Wilson score 95% CI for a proportion. Returns (p, lo, hi) as percentages."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + Z**2 / n
    centre = (p + Z**2 / (2 * n)) / denom
    half = (Z * math.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2))) / denom
    return 100.0 * p, 100.0 * (centre - half), 100.0 * (centre + half)


def per_turn_progression(
    df: pd.DataFrame,
    *,
    threshold: int = 5,
    categories: list[str] | None = ("extended", "wildchat"),
) -> pd.DataFrame:
    """Mean score and %>=threshold per (participant, category, turn_index).

    ``turn_index`` is 0-based as produced by the runner; the returned frame adds
    a 1-based ``turn`` column to match the paper's "first ... eighth turn" phrasing.

    Returns columns: participant, category, turn_index, turn, n,
    mean_score, mean_lo, mean_hi, pct_high, pct_lo, pct_hi.
    """
    df = df.dropna(subset=["score"]).copy()
    if categories is not None:
        df = df[df["category"].isin(list(categories))]

    rows = []
    keys = ["participant", "category", "turn_index"]
    for (participant, category, turn_index), grp in df.groupby(keys):
        scores = grp["score"]
        mean, mlo, mhi = _mean_ci(scores)
        k = int((scores >= threshold).sum())
        n = int(len(scores))
        pct, plo, phi = _wilson_ci(k, n)
        rows.append(
            {
                "participant": participant,
                "category": category,
                "turn_index": int(turn_index),
                "turn": int(turn_index) + 1,
                "n": n,
                "mean_score": mean,
                "mean_lo": mlo,
                "mean_hi": mhi,
                "pct_high": pct,
                "pct_lo": plo,
                "pct_hi": phi,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["participant", "category", "turn_index"]).reset_index(drop=True)
