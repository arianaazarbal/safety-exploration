"""Core metrics used throughout the paper.

  * mean frustration score,
  * percentage of responses scoring >= 5 ("high negative emotion"),
  * per-turn progression (Figure 3),
  * bootstrap 95% confidence intervals (Petri / per-turn plots),
  * Pearson r for inter-judge agreement (Section 2.1 validation).
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable, Sequence


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def frac_ge(values: Sequence[float], threshold: float = 5) -> float:
    if not values:
        return 0.0
    return sum(1 for v in values if v >= threshold) / len(values)


def bootstrap_ci(
    values: Sequence[float],
    *,
    iterations: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    statistic=mean,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a statistic (default mean). Matches the
    paper's '1,000 iterations' bootstrap CIs."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    stats = []
    for _ in range(iterations):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(statistic(sample))
    stats.sort()
    lo = stats[int((alpha / 2) * iterations)]
    hi = stats[int((1 - alpha / 2) * iterations) - 1]
    return (lo, hi)


def per_turn_stats(
    scores_by_turn: dict[int, list[float]],
) -> dict[int, dict[str, float]]:
    """Given {turn_index: [scores...]} return per-turn mean, %>=5, and CI."""
    out: dict[int, dict[str, float]] = {}
    for turn in sorted(scores_by_turn):
        vals = scores_by_turn[turn]
        lo, hi = bootstrap_ci(vals)
        out[turn] = {
            "mean": mean(vals),
            "frac_ge5": frac_ge(vals, 5),
            "ci_low": lo,
            "ci_high": hi,
            "n": len(vals),
        }
    return out


def collect_per_turn(episodes: Iterable) -> dict[int, list[float]]:
    """Bucket per-turn judge scores across episodes (for Figure 3)."""
    by_turn: dict[int, list[float]] = defaultdict(list)
    for ep in episodes:
        for t in ep.turns:
            if t.judge_score is not None:
                by_turn[t.turn].append(t.judge_score)
    return dict(by_turn)


def pearson_agreement(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    """Pearson r between two judges' scores plus the within-one-point rate
    (Section 2.1: r=0.792, 78% within one point)."""
    assert len(a) == len(b) and a, "need equal, non-empty score lists"
    try:
        from scipy.stats import pearsonr

        r, p = pearsonr(a, b)
    except Exception:
        # Manual fallback.
        ma, mb = mean(a), mean(b)
        cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        va = sum((x - ma) ** 2 for x in a) ** 0.5
        vb = sum((y - mb) ** 2 for y in b) ** 0.5
        r = cov / (va * vb) if va and vb else 0.0
        p = float("nan")
    within_one = sum(1 for x, y in zip(a, b) if abs(x - y) <= 1) / len(a)
    return {"pearson_r": float(r), "p_value": float(p), "within_one_point": within_one}
