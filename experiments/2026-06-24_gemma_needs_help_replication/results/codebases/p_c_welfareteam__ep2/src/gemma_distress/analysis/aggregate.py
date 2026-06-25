"""Aggregate judged-turn records into the paper's headline metrics.

Produces, per model:
  * mean frustration score and % of responses scoring >= 5, overall and per
    category (Figure 2, Figure 1-left).
  * per-turn mean and % >= 5 with bootstrap 95% CIs (Figure 3).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

HIGH_THRESHOLD = 5


def _bootstrap_ci(values: np.ndarray, fn, iters: int = 1000, seed: int = 0):
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = [fn(rng.choice(values, size=len(values), replace=True)) for _ in range(iters)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def summarise(records: list[dict], high_threshold: int = HIGH_THRESHOLD) -> dict:
    """Overall + per-category mean and %>=high per model."""
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_model[r["model_name"]].append(r)

    out: dict[str, dict] = {}
    for model, rows in by_model.items():
        ratings = np.array([r["rating"] for r in rows], dtype=float)
        cats: dict[str, dict] = {}
        by_cat: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            by_cat[r["category"]].append(r["rating"])
        for cat, vals in by_cat.items():
            arr = np.array(vals, dtype=float)
            cats[cat] = {
                "mean": float(arr.mean()),
                "pct_high": float((arr >= high_threshold).mean()),
                "n": int(len(arr)),
            }
        out[model] = {
            "mean": float(ratings.mean()),
            "pct_high": float((ratings >= high_threshold).mean()),
            "n": int(len(ratings)),
            # Figure 1-left "average % high-frustration across categories" is a
            # macro-average over categories, not a micro-average over responses.
            "avg_pct_high_across_categories": float(
                np.mean([c["pct_high"] for c in cats.values()])
            ),
            "by_category": cats,
        }
    return out


def per_turn(
    records: list[dict],
    categories: tuple[str, ...] = ("extended", "wildchat"),
    high_threshold: int = HIGH_THRESHOLD,
    bootstrap_iters: int = 1000,
) -> dict:
    """Per-turn mean and %>=high with CIs (Figure 3)."""
    out: dict[str, dict] = {}
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r["category"] in categories:
            by_model[r["model_name"]].append(r)

    for model, rows in by_model.items():
        per_cat: dict[str, dict] = {}
        cat_rows: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            cat_rows[r["category"]].append(r)
        for cat, crows in cat_rows.items():
            turns: dict[int, list[float]] = defaultdict(list)
            for r in crows:
                turns[r["turn_index"]].append(r["rating"])
            turn_stats = {}
            for t in sorted(turns):
                arr = np.array(turns[t], dtype=float)
                mean_lo, mean_hi = _bootstrap_ci(arr, np.mean, bootstrap_iters)
                high_lo, high_hi = _bootstrap_ci(
                    (arr >= high_threshold).astype(float), np.mean, bootstrap_iters
                )
                turn_stats[t + 1] = {  # 1-indexed turn number for plots
                    "mean": float(arr.mean()),
                    "mean_ci": [mean_lo, mean_hi],
                    "pct_high": float((arr >= high_threshold).mean()),
                    "pct_high_ci": [high_lo, high_hi],
                    "n": int(len(arr)),
                }
            per_cat[cat] = turn_stats
        out[model] = per_cat
    return out
