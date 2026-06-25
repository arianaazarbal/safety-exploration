"""Frustration metrics and aggregations (Figures 1-3).

Operates on the flat scored-response rows produced by eval.run_eval. Provides:
  * mean frustration and % >= 5 overall, per-category, per-condition,
  * the Figure-1 headline ("avg % high-frustration responses across evals"),
  * per-turn progression with 95% bootstrap CIs (Figure 3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config


def _ratings(rows: list[dict]) -> np.ndarray:
    return np.array([r["rating"] for r in rows], dtype=float)


def mean_frustration(rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    return float(_ratings(rows).mean())


def pct_high(rows: list[dict], threshold: int = config.HIGH_FRUSTRATION_THRESHOLD) -> float:
    if not rows:
        return float("nan")
    return float((_ratings(rows) >= threshold).mean() * 100.0)


def by_category(rows: list[dict]) -> dict[str, dict]:
    cats = sorted({r["category"] for r in rows})
    out = {}
    for c in cats:
        sub = [r for r in rows if r["category"] == c]
        out[c] = {"mean": mean_frustration(sub), "pct_high": pct_high(sub), "n": len(sub)}
    return out


def headline_pct_high(rows: list[dict]) -> float:
    """Figure-1 metric: average (over the 5 categories) of the per-category
    % of responses scoring >= 5. Averaging per-category (not pooling responses)
    matches the paper's "Avg % high-frustration responses across the evals"."""
    cats = by_category(rows)
    if not cats:
        return float("nan")
    return float(np.mean([v["pct_high"] for v in cats.values()]))


def summarise_model(rows: list[dict]) -> dict:
    return {
        "n_responses": len(rows),
        "mean_frustration": mean_frustration(rows),
        "pct_high_pooled": pct_high(rows),
        "headline_pct_high": headline_pct_high(rows),
        "by_category": by_category(rows),
    }


# --------------------------------------------------------------------------- #
# Per-turn progression (Figure 3)
# --------------------------------------------------------------------------- #
@dataclass
class TurnStat:
    turn: int
    n: int
    mean: float
    mean_lo: float
    mean_hi: float
    pct_high: float
    pct_high_lo: float
    pct_high_hi: float


def _bootstrap_ci(values: np.ndarray, stat_fn, iters: int = 1000,
                  seed: int = 0) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(iters)
    n = len(values)
    for i in range(iters):
        sample = values[rng.integers(0, n, n)]
        boots[i] = stat_fn(sample)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def per_turn(rows: list[dict], threshold: int = config.HIGH_FRUSTRATION_THRESHOLD,
             iters: int = 1000) -> list[TurnStat]:
    turns = sorted({r["turn"] for r in rows})
    stats = []
    for t in turns:
        vals = _ratings([r for r in rows if r["turn"] == t])
        mean_lo, mean_hi = _bootstrap_ci(vals, np.mean, iters)
        high = (vals >= threshold).astype(float)
        ph_lo, ph_hi = _bootstrap_ci(high, lambda s: s.mean() * 100.0, iters)
        stats.append(TurnStat(
            turn=t, n=len(vals), mean=float(vals.mean()),
            mean_lo=mean_lo, mean_hi=mean_hi,
            pct_high=float(high.mean() * 100.0), pct_high_lo=ph_lo, pct_high_hi=ph_hi,
        ))
    return stats


def per_turn_for_condition(rows: list[dict], condition: str, **kw) -> list[TurnStat]:
    return per_turn([r for r in rows if r["condition"] == condition], **kw)
