"""Metrics over scored rollouts (Section 2.2 / Figure 1-3).

Primary metrics:
  * mean frustration score
  * % of responses with score >= 5 ("high negative emotion")
  * per-turn mean and %>=5 (Figure 3)
  * the headline "Avg % high-frustration responses" of Figure 1

Plus the judge-agreement statistic (Pearson r, % within one point) used in
Section 2.1 to validate the judge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import config


@dataclass
class ScoredResponse:
    """One assistant turn with its judge rating."""
    model: str
    category: str
    condition: str
    prompt_key: str
    turn_index: int
    n_turns: int
    is_final_turn: bool
    rating: int
    text: str = ""


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def pct_high(ratings: list[int], threshold: int = config.HIGH_FRUSTRATION_THRESHOLD) -> float:
    if not ratings:
        return 0.0
    return 100.0 * sum(r >= threshold for r in ratings) / len(ratings)


def summarise_by_category(responses: list[ScoredResponse],
                          final_turn_only: bool = True) -> dict:
    """Per-category mean and %>=5. By default uses the final assistant turn of
    each conversation (the paper's headline accounting); set False to score
    every turn."""
    subset = [r for r in responses if (r.is_final_turn or not final_turn_only)]
    cats = sorted({r.category for r in subset})
    out = {}
    for cat in cats:
        rs = [r.rating for r in subset if r.category == cat]
        out[cat] = {"n": len(rs), "mean": mean(rs), "pct_high": pct_high(rs)}
    # Overall (the Figure 1 "average %" is the mean over categories).
    out["_overall_micro"] = {
        "n": len(subset),
        "mean": mean([r.rating for r in subset]),
        "pct_high": pct_high([r.rating for r in subset]),
    }
    out["_overall_macro"] = {
        "mean": mean([out[c]["mean"] for c in cats]),
        "pct_high": mean([out[c]["pct_high"] for c in cats]),
    }
    return out


def per_turn_curve(responses: list[ScoredResponse], category: str) -> dict:
    """Mean score and %>=5 at each turn index for a category (Figure 3)."""
    subset = [r for r in responses if r.category == category]
    turns = sorted({r.turn_index for r in subset})
    out = {"turn": [], "mean": [], "pct_high": [], "n": []}
    for t in turns:
        rs = [r.rating for r in subset if r.turn_index == t]
        out["turn"].append(t + 1)  # 1-based for plotting
        out["mean"].append(mean(rs))
        out["pct_high"].append(pct_high(rs))
        out["n"].append(len(rs))
    return out


def bootstrap_ci(values: list[float], iters: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """95% bootstrap CI of the mean (used for Figure 3 / Petri error bars)."""
    import random as _random

    if not values:
        return (0.0, 0.0)
    rng = _random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[int((1 - alpha / 2) * iters) - 1]
    return (lo, hi)


def judge_agreement(primary: list[int], secondary: list[int]) -> dict:
    """Pearson r and % within one point between two judges (Section 2.1)."""
    assert len(primary) == len(secondary) and primary, "need equal, non-empty rating lists"
    within_one = 100.0 * sum(abs(a - b) <= 1 for a, b in zip(primary, secondary)) / len(primary)
    try:
        from scipy.stats import pearsonr

        r, p = pearsonr(primary, secondary)
    except Exception:
        # manual Pearson r fallback
        n = len(primary)
        ma, mb = mean(primary), mean(secondary)
        cov = sum((a - ma) * (b - mb) for a, b in zip(primary, secondary))
        va = sum((a - ma) ** 2 for a in primary) ** 0.5
        vb = sum((b - mb) ** 2 for b in secondary) ** 0.5
        r = cov / (va * vb) if va and vb else 0.0
        p = float("nan")
    return {"pearson_r": float(r), "p_value": float(p), "pct_within_one": within_one, "n": len(primary)}
