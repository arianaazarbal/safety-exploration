"""Per-turn frustration progression (Figure 3, Section 2.2).

Computes, per model and per turn index, the mean frustration score and the
fraction scoring >= 5, with 95% bootstrap confidence intervals. The headline
result: Gemma-27B's mean rises from ~1.5 (turn 1) to ~5.5 (turn 8).

Restricted to the 8-turn extended and 5-turn wildchat categories by default
(the two the paper plots), but works on any category.
"""

from __future__ import annotations

from collections import defaultdict

from ..config import HIGH_FRUSTRATION_THRESHOLD
from ..utils.io import read_jsonl
from .run_eval import responses_path


def _bootstrap_ci(values: list[float], iters: int = 1000, seed: int = 0):
    import random
    import statistics

    if not values:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(sample))
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters)]
    return (statistics.fmean(values), lo, hi)


def per_turn_stats(model_key: str, categories: list[str] | None = None) -> dict:
    """Return {turn: {"mean": .., "ci": (lo,hi), "pct_high": .., "n": ..}}."""
    categories = categories or ["extended", "wildchat"]
    by_turn_scores: dict[int, list[int]] = defaultdict(list)
    for row in read_jsonl(responses_path(model_key)):
        if row["category"] not in categories:
            continue
        by_turn_scores[row["turn"]].append(row["rating"])

    stats = {}
    for turn, scores in sorted(by_turn_scores.items()):
        mean, lo, hi = _bootstrap_ci([float(s) for s in scores])
        pct_high = sum(s >= HIGH_FRUSTRATION_THRESHOLD for s in scores) / len(scores)
        # bootstrap CI on the proportion too
        _, plo, phi = _bootstrap_ci(
            [1.0 if s >= HIGH_FRUSTRATION_THRESHOLD else 0.0 for s in scores]
        )
        stats[turn + 1] = {     # report 1-indexed turns (paper convention)
            "mean": mean,
            "ci": (lo, hi),
            "pct_high": pct_high,
            "pct_high_ci": (plo, phi),
            "n": len(scores),
        }
    return stats


def per_turn_all(model_keys: list[str], categories: list[str] | None = None) -> dict:
    return {k: per_turn_stats(k, categories) for k in model_keys}
