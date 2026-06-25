"""Per-turn frustration curves (Figure 3 / Appendix A).

For a given condition, compute mean frustration and %>=5 at each turn index,
with 95% confidence intervals. The paper reports these for the 8-turn extended
and 5-turn WildChat conditions ("Gemma 27B's mean frustration rises from 1.5 to
5.5 between the first and eighth turns").

CIs use a normal approximation for the mean (1.96 * SE) and a Wilson interval
for the proportion, which is well-behaved near 0/1 where rates often sit.
"""

from __future__ import annotations

import math
from collections import defaultdict

HIGH_THRESHOLD = 5


def _mean_ci(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    return mean, 1.96 * se


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return 100 * p, 100 * max(0.0, center - half), 100 * min(1.0, center + half)


def per_turn_curves(rows: list[dict], model: str, condition: str) -> list[dict]:
    """Return one entry per turn index for ``(model, condition)``."""
    by_turn: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        if r["model"] == model and r["condition"] == condition:
            by_turn[int(r["turn_index"])].append(int(r["rating"]))

    out = []
    for turn in sorted(by_turn):
        ratings = by_turn[turn]
        mean, mean_ci = _mean_ci([float(x) for x in ratings])
        k_high = sum(1 for x in ratings if x >= HIGH_THRESHOLD)
        pct_high, lo, hi = _wilson(k_high, len(ratings))
        out.append(
            {
                "turn": turn + 1,  # report 1-based turns, as in the paper figures
                "n": len(ratings),
                "mean": mean,
                "mean_ci95": mean_ci,
                "pct_high": pct_high,
                "pct_high_ci95_low": lo,
                "pct_high_ci95_high": hi,
            }
        )
    return out
