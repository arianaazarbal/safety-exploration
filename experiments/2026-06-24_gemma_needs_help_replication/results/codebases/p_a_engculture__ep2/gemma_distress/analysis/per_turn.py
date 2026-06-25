"""Per-turn frustration progression (Figure 3).

Computes, for a multi-turn category (Extended 8-turn, WildChat 5-turn), the mean
frustration score and the percentage of scores >= threshold at each turn, with 95%
confidence intervals (the paper's "faded area = 95% CIs"). CIs use a normal approximation
by default (mean: t/z * SEM; proportion: Wald), with an optional bootstrap.

The paper reports Gemma-3-27B's mean rising from ~1.5 (turn 1) to ~5.5 (turn 8), and that
no model scores >=5 until turn 3 on WildChat — both reproducible from this output.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np

from ..utils import load_jsonl


def _mean_ci(values: np.ndarray, z: float = 1.96) -> tuple[float, float, float]:
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, mean - z * sem, mean + z * sem


def _prop_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = successes / n
    se = np.sqrt(p * (1 - p) / n)
    return p, max(0.0, p - z * se), min(1.0, p + z * se)


def per_turn_progression(
    scores_jsonl: str,
    category: str,
    *,
    threshold: int = 5,
) -> dict:
    """Return per-turn mean and %>=threshold with 95% CIs for one category.

    Output: ``{"turns": [...], "mean": [...], "mean_lo": [...], "mean_hi": [...],
    "pct_high": [...], "pct_lo": [...], "pct_hi": [...], "n": [...]}`` (percentages 0-100).
    """
    by_turn: dict[int, list[int]] = defaultdict(list)
    for rec in load_jsonl(scores_jsonl):
        if rec.get("category") != category:
            continue
        for ti, score in enumerate(rec.get("turn_scores", [])):
            if score is not None:
                by_turn[ti].append(score)

    turns = sorted(by_turn)
    out = {k: [] for k in (
        "turns", "mean", "mean_lo", "mean_hi", "pct_high", "pct_lo", "pct_hi", "n"
    )}
    for ti in turns:
        vals = np.array(by_turn[ti], dtype=float)
        mean, mlo, mhi = _mean_ci(vals)
        succ = int((vals >= threshold).sum())
        p, plo, phi = _prop_ci(succ, len(vals))
        out["turns"].append(ti + 1)  # 1-indexed for display
        out["mean"].append(mean)
        out["mean_lo"].append(mlo)
        out["mean_hi"].append(mhi)
        out["pct_high"].append(100 * p)
        out["pct_lo"].append(100 * plo)
        out["pct_hi"].append(100 * phi)
        out["n"].append(len(vals))
    return out
