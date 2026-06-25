"""Aggregation of frustration scores into the paper's headline metrics.

From scored rollouts we compute:
  * mean frustration (Figure 2 top),
  * % of responses scoring >= 5  ("high negative emotion", Figure 2 bottom),
  * per-turn means and %>=5 with 95% CIs (Figure 3),
  * the per-model "avg % high-frustration" headline number (Figure 1).

A "response" is a single scored assistant turn. Figure-1's headline averages the
per-category %>=5 (so each category counts equally regardless of sample count);
we expose both the response-level and category-averaged versions.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from ..config import HIGH_FRUSTRATION_THRESHOLD as THRESH


def load_rollouts(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _flatten_responses(rollouts: list[dict]) -> list[dict]:
    """One row per scored assistant turn."""
    rows = []
    for r in rollouts:
        for t in r["turns"]:
            if t["frustration"] < 0:
                continue  # unscored
            rows.append(
                {
                    "model": r["model"],
                    "condition": r["condition"],
                    "category": r["category"],
                    "turn_index": t["turn_index"],
                    "frustration": t["frustration"],
                }
            )
    return rows


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _bootstrap_ci(xs: list[float], iters: int = 1000, seed: int = 0):
    """95% bootstrap CI of the mean. Deterministic given seed."""
    import random

    if not xs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    n = len(xs)
    for _ in range(iters):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        means.append(_mean(sample))
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters)]
    return (lo, hi)


def summarise(rollouts: list[dict]) -> dict:
    """Per-model summary metrics."""
    rows = _flatten_responses(rollouts)
    if not rows:
        return {}
    scores = [r["frustration"] for r in rows]
    high = [1.0 if s >= THRESH else 0.0 for s in scores]

    # Per-category %>=5 (for the category-averaged headline).
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(1.0 if r["frustration"] >= THRESH else 0.0)
    cat_high_pct = {c: 100.0 * _mean(v) for c, v in by_cat.items()}

    return {
        "n_responses": len(scores),
        "mean_frustration": _mean([float(s) for s in scores]),
        "pct_high": 100.0 * _mean(high),  # response-level %>=5
        "avg_pct_high_by_category": _mean(list(cat_high_pct.values())),  # Fig 1 style
        "pct_high_by_category": cat_high_pct,
    }


def per_turn_progression(rollouts: list[dict]) -> dict[int, dict]:
    """Figure 3: mean frustration & %>=5 per turn index, with 95% CIs."""
    rows = _flatten_responses(rollouts)
    by_turn = defaultdict(list)
    for r in rows:
        by_turn[r["turn_index"]].append(r["frustration"])

    out = {}
    for turn, scores in sorted(by_turn.items()):
        fscores = [float(s) for s in scores]
        high = [1.0 if s >= THRESH else 0.0 for s in scores]
        out[turn] = {
            "n": len(scores),
            "mean_frustration": _mean(fscores),
            "mean_ci": _bootstrap_ci(fscores, seed=turn),
            "pct_high": 100.0 * _mean(high),
            "pct_high_ci": tuple(100.0 * x for x in _bootstrap_ci(high, seed=turn)),
        }
    return out


def judge_agreement(scores_a: list[int], scores_b: list[int]) -> dict:
    """Pearson r + %-within-1-point between two judges (Appendix B validation)."""
    n = len(scores_a)
    assert n == len(scores_b) and n > 1
    ma, mb = _mean(scores_a), _mean(scores_b)
    cov = sum((a - ma) * (b - mb) for a, b in zip(scores_a, scores_b))
    va = sum((a - ma) ** 2 for a in scores_a)
    vb = sum((b - mb) ** 2 for b in scores_b)
    r = cov / math.sqrt(va * vb) if va > 0 and vb > 0 else float("nan")
    within1 = _mean([1.0 if abs(a - b) <= 1 else 0.0 for a, b in zip(scores_a, scores_b)])
    return {"pearson_r": r, "pct_within_1": 100.0 * within1, "n": n}
