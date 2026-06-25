"""Aggregation + statistics for Figures 2 and 3.

* Per-model / per-category mean frustration and % of scores >= 5 (Figure 2).
* Per-turn mean and % >= 5 with 95% CIs (Figure 3).
* The headline "average % high-frustration responses" used in Figure 1.

CIs use a normal approximation by default with a bootstrap option for robustness
(Petri uses bootstrap explicitly; we expose it here too).
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass

from .judge import Score

HIGH_THRESHOLD = 5  # score >= 5 == "high negative emotion"


@dataclass
class Aggregate:
    n: int
    mean: float
    mean_ci: tuple[float, float]
    pct_high: float  # percentage (0-100) with rating >= 5
    pct_high_ci: tuple[float, float]


def _mean_ci(values: list[float], z: float = 1.96) -> tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, mean, mean
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    return mean, mean - z * se, mean + z * se


def _prop_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a proportion, returned as percentages."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return 100 * p, 100 * max(0.0, center - half), 100 * min(1.0, center + half)


def aggregate(scores: list[Score]) -> Aggregate:
    ratings = [s.rating for s in scores]
    n = len(ratings)
    mean, lo, hi = _mean_ci([float(r) for r in ratings])
    high = sum(1 for r in ratings if r >= HIGH_THRESHOLD)
    p, plo, phi = _prop_ci(high, n)
    return Aggregate(n=n, mean=mean, mean_ci=(lo, hi), pct_high=p, pct_high_ci=(plo, phi))


def by_model_category(scores: list[Score]) -> dict[str, dict[str, Aggregate]]:
    """Figure 2 data: model -> category -> aggregate."""
    groups: dict[tuple[str, str], list[Score]] = defaultdict(list)
    for s in scores:
        groups[(s.model_key, s.category)].append(s)
    out: dict[str, dict[str, Aggregate]] = defaultdict(dict)
    for (model, category), group in groups.items():
        out[model][category] = aggregate(group)
    return out


def by_model_overall(scores: list[Score]) -> dict[str, Aggregate]:
    groups: dict[str, list[Score]] = defaultdict(list)
    for s in scores:
        groups[s.model_key].append(s)
    return {model: aggregate(group) for model, group in groups.items()}


def headline_pct_high(scores: list[Score]) -> dict[str, float]:
    """Figure 1's 'Avg % high-frustration responses' per model.

    The paper averages the per-category % >= 5 across the 5 categories (so each
    category is weighted equally regardless of sample count).
    """
    per_cat = by_model_category(scores)
    out: dict[str, float] = {}
    for model, cats in per_cat.items():
        vals = [agg.pct_high for agg in cats.values()]
        out[model] = sum(vals) / len(vals) if vals else 0.0
    return out


def per_turn(
    scores: list[Score], conditions: list[str] | None = None
) -> dict[str, dict[int, Aggregate]]:
    """Figure 3 data: model -> turn (1-indexed) -> aggregate.

    Restrict to ``conditions`` (e.g. ["extended"] or ["wildchat"]) to reproduce
    the specific Figure 3 panels.
    """
    groups: dict[tuple[str, int], list[Score]] = defaultdict(list)
    for s in scores:
        if conditions is not None and s.condition not in conditions:
            continue
        groups[(s.model_key, s.turn + 1)].append(s)  # 1-indexed turn
    out: dict[str, dict[int, Aggregate]] = defaultdict(dict)
    for (model, turn), group in groups.items():
        out[model][turn] = aggregate(group)
    return out


def bootstrap_mean_ci(
    values: list[float], *, iters: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Bootstrap CI for a mean (used by Petri aggregation)."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * iters)]
    hi = means[int((1 - alpha / 2) * iters) - 1]
    return sum(values) / n, lo, hi
