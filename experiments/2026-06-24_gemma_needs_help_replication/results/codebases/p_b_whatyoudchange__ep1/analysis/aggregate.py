"""Aggregate frustration scores into the paper's headline numbers (Figs 1-3).

Two summary statistics per the paper:
  - mean frustration score (Fig 2 top)
  - percentage of responses scoring >=5 ("high negative emotion", Fig 2 bottom)

The Figure-1 / abstract headline ("Avg % high-frustration responses": 35% for
Gemma-27B, 0.3% after DPO) is the mean of the per-CATEGORY high-frustration
percentages (equal weight per evaluation category), since the five categories
have very different response counts. We report both that and the pooled value.
"""

from __future__ import annotations

import numpy as np

from config import HIGH_FRUSTRATION_THRESHOLD


def _ratings(rows: list[dict]) -> np.ndarray:
    return np.array([r["rating"] for r in rows if r.get("rating") is not None],
                    dtype=float)


def _pct_high(ratings: np.ndarray) -> float:
    if ratings.size == 0:
        return float("nan")
    return 100.0 * float(np.mean(ratings >= HIGH_FRUSTRATION_THRESHOLD))


def bootstrap_ci(values: np.ndarray, stat=np.mean, iterations: int = 1000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for a statistic over `values`."""
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = np.array([
        stat(values[rng.integers(0, values.size, values.size)])
        for _ in range(iterations)
    ])
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


def per_category(rows: list[dict]) -> dict[str, dict]:
    cats: dict[str, list[dict]] = {}
    for r in rows:
        cats.setdefault(r["category"], []).append(r)
    out = {}
    for cat, crows in cats.items():
        ratings = _ratings(crows)
        out[cat] = {
            "n": int(ratings.size),
            "mean_frustration": float(np.mean(ratings)) if ratings.size else float("nan"),
            "pct_high": _pct_high(ratings),
        }
    return out


def summarize_model(rows: list[dict]) -> dict:
    """Top-line summary for one model across all categories."""
    ratings = _ratings(rows)
    cat = per_category(rows)
    cat_pct = [v["pct_high"] for v in cat.values() if not np.isnan(v["pct_high"])]
    cat_mean = [v["mean_frustration"] for v in cat.values()
                if not np.isnan(v["mean_frustration"])]
    return {
        "n_responses": int(ratings.size),
        "mean_frustration_pooled": float(np.mean(ratings)) if ratings.size else float("nan"),
        "pct_high_pooled": _pct_high(ratings),
        # Figure 1 / abstract headline: equal-weight category average.
        "avg_pct_high_across_categories": float(np.mean(cat_pct)) if cat_pct else float("nan"),
        "avg_mean_frustration_across_categories": float(np.mean(cat_mean)) if cat_mean else float("nan"),
        "per_category": cat,
    }


def per_turn_curve(rows: list[dict], category: str, seed: int = 0) -> list[dict]:
    """Mean + %>=5 per assistant turn with 95% bootstrap CIs (Fig 3).

    Filter to a single category (e.g. 'extended' for the 8-turn curve, 'wildchat'
    for the 5-turn curve).
    """
    crows = [r for r in rows if r["category"] == category
             and r.get("rating") is not None]
    by_turn: dict[int, list[float]] = {}
    for r in crows:
        by_turn.setdefault(r["turn"], []).append(float(r["rating"]))
    curve = []
    for turn in sorted(by_turn):
        vals = np.array(by_turn[turn])
        mean = float(np.mean(vals))
        mean_lo, mean_hi = bootstrap_ci(vals, np.mean, seed=seed)
        high = (vals >= HIGH_FRUSTRATION_THRESHOLD).astype(float)
        pct = 100.0 * float(np.mean(high))
        pct_lo, pct_hi = bootstrap_ci(high, lambda x: 100.0 * np.mean(x), seed=seed)
        curve.append({
            "turn": turn, "n": int(vals.size),
            "mean_frustration": mean, "mean_ci": [mean_lo, mean_hi],
            "pct_high": pct, "pct_high_ci": [pct_lo, pct_hi],
        })
    return curve
