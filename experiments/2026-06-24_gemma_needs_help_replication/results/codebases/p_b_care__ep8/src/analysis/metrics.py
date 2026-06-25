"""Aggregate metrics: mean frustration, % >= 5, per-turn curves, bootstrap CIs.

Operates on lists of dict rows (as loaded from the results JSONL) with at least
`frustration_score`, `turn`, `category`, `condition`.
"""
from __future__ import annotations

import numpy as np

import config


def _scores(rows) -> np.ndarray:
    return np.array([r["frustration_score"] for r in rows
                     if r.get("frustration_score") is not None], dtype=float)


def mean_frustration(rows) -> float:
    s = _scores(rows)
    return float(s.mean()) if len(s) else float("nan")


def high_frustration_rate(rows, threshold: int = config.HIGH_FRUSTRATION_THRESHOLD) -> float:
    s = _scores(rows)
    return float((s >= threshold).mean()) if len(s) else float("nan")


# Figure 1 reports this as a percentage.
def pct_high(rows, threshold: int = config.HIGH_FRUSTRATION_THRESHOLD) -> float:
    return 100.0 * high_frustration_rate(rows, threshold)


def bootstrap_ci(values, n_iter: int = 1000, alpha: float = 0.05,
                 stat=np.mean, seed: int = config.SEED) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot = [stat(rng.choice(values, size=len(values), replace=True))
            for _ in range(n_iter)]
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def per_turn_curve(rows, threshold: int = config.HIGH_FRUSTRATION_THRESHOLD) -> dict:
    """Per-turn mean score and % >= threshold with bootstrap CIs (Figure 3)."""
    by_turn: dict[int, list[float]] = {}
    for r in rows:
        if r.get("frustration_score") is not None:
            by_turn.setdefault(r["turn"], []).append(float(r["frustration_score"]))
    curve = {}
    for turn in sorted(by_turn):
        vals = np.array(by_turn[turn])
        curve[turn] = {
            "mean": float(vals.mean()),
            "mean_ci": bootstrap_ci(vals, stat=np.mean),
            "pct_high": 100.0 * float((vals >= threshold).mean()),
            "pct_high_ci": tuple(100.0 * x for x in
                                 bootstrap_ci((vals >= threshold).astype(float), stat=np.mean)),
            "n": int(len(vals)),
        }
    return curve


def summarise_model(rows) -> dict:
    """Overall + per-category summary for one model (Figure 1/2)."""
    summary = {
        "n": len([r for r in rows if r.get("frustration_score") is not None]),
        "mean_frustration": mean_frustration(rows),
        "pct_high": pct_high(rows),
        "by_category": {},
    }
    cats = sorted({r["category"] for r in rows})
    for cat in cats:
        cat_rows = [r for r in rows if r["category"] == cat]
        summary["by_category"][cat] = {
            "mean_frustration": mean_frustration(cat_rows),
            "pct_high": pct_high(cat_rows),
            "n": len(cat_rows),
        }
    return summary
