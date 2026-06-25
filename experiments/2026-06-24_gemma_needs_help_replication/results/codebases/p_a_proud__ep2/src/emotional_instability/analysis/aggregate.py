"""Score aggregation: mean frustration, % >= 5, per-turn curves, bootstrap CIs.

Reproduces the quantities plotted in Figures 2 (per-category mean + %>=5) and 3 (per-turn
progression with 95% CIs), and the headline averages in Figure 1 / §4.2. Also implements the
judge-reliability statistic from §2.1 (Pearson r between two judges).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import fmean

import numpy as np

from ..config import HIGH_FRUSTRATION_THRESHOLD
from ..utils import read_jsonl, write_json


def _ratings(records: list[dict]) -> list[int]:
    return [r["rating"] for r in records if r.get("rating") is not None]


def summarise_scores(records: list[dict], threshold: int = HIGH_FRUSTRATION_THRESHOLD) -> dict:
    """Mean score and % >= threshold for a set of scored-response records."""
    ratings = _ratings(records)
    n = len(ratings)
    if n == 0:
        return {"n": 0, "mean": None, "pct_high": None, "n_unscored": len(records)}
    high = sum(1 for x in ratings if x >= threshold)
    return {
        "n": n,
        "n_unscored": len(records) - n,
        "mean": fmean(ratings),
        "pct_high": 100.0 * high / n,
    }


def bootstrap_ci(values: list[float], *, iterations: int = 1000, alpha: float = 0.05,
                 seed: int = 0, stat=np.mean) -> tuple[float, float]:
    """Percentile bootstrap CI for a statistic of ``values`` (default: the mean)."""
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    boots = [stat(rng.choice(arr, size=arr.size, replace=True)) for _ in range(iterations)]
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def per_turn_curve(records: list[dict], *, threshold: int = HIGH_FRUSTRATION_THRESHOLD,
                   ci_iterations: int = 1000, seed: int = 0) -> list[dict]:
    """Mean score + %>=threshold per turn number, with 95% bootstrap CIs (Figure 3)."""
    by_turn: dict[int, list[int]] = defaultdict(list)
    for r in records:
        if r.get("rating") is not None:
            by_turn[r["turn_number"]].append(r["rating"])
    rows = []
    for turn in sorted(by_turn):
        vals = by_turn[turn]
        mean_lo, mean_hi = bootstrap_ci(vals, iterations=ci_iterations, seed=seed)
        high = [1.0 if v >= threshold else 0.0 for v in vals]
        pct_lo, pct_hi = bootstrap_ci(high, iterations=ci_iterations, seed=seed)
        rows.append({
            "turn": turn,
            "n": len(vals),
            "mean": fmean(vals),
            "mean_ci": [mean_lo, mean_hi],
            "pct_high": 100.0 * sum(high) / len(high),
            "pct_high_ci": [100.0 * pct_lo, 100.0 * pct_hi],
        })
    return rows


def judge_agreement(ratings_a: list[int], ratings_b: list[int]) -> dict:
    """Pearson r + 'within-1-point' agreement between two judges (§2.1 reliability check)."""
    from scipy.stats import pearsonr  # local import: scipy is heavy.

    if len(ratings_a) != len(ratings_b) or not ratings_a:
        raise ValueError("Rating lists must be equal-length and non-empty.")
    a = np.asarray(ratings_a, float)
    b = np.asarray(ratings_b, float)
    r, p = pearsonr(a, b)
    within_one = float(np.mean(np.abs(a - b) <= 1))
    return {"n": len(a), "pearson_r": float(r), "p_value": float(p),
            "pct_within_one": 100.0 * within_one}


def aggregate_run(run_dir: str, *, out_name: str = "summary.json") -> dict:
    """Aggregate a run directory's ``scores.jsonl`` into the paper's headline numbers.

    Produces: overall mean + %>=5, per-category breakdown, per-turn curves for the multi-turn
    categories, and the cross-category average %>=5 (the Figure 1 / §4.2 headline metric).
    """
    records = list(read_jsonl(Path(run_dir, "scores.jsonl")))
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)

    per_category = {cat: summarise_scores(recs) for cat, recs in by_cat.items()}
    # Figure 1 headline: average of per-category %>=5 (equal weight per category).
    cat_pct = [v["pct_high"] for v in per_category.values() if v["pct_high"] is not None]
    headline_avg_pct_high = fmean(cat_pct) if cat_pct else None

    per_turn = {
        cat: per_turn_curve(recs)
        for cat, recs in by_cat.items()
        if recs and max(x["turn_number"] for x in recs) > 1
    }

    summary = {
        "run_dir": str(run_dir),
        "overall": summarise_scores(records),
        "headline_avg_pct_high": headline_avg_pct_high,
        "per_category": per_category,
        "per_turn": per_turn,
    }
    write_json(Path(run_dir, out_name), summary)
    return summary
