"""Aggregation of scored rollouts into the paper's headline metrics.

Produces:
  * mean frustration and % of responses scoring >= threshold, overall and per
    category (Figure 1, Figure 2)
  * per-turn mean and %>=threshold with 95% bootstrap CIs (Figure 3)
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path


def load_rollouts(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _all_scores(rollouts: list[dict]) -> list[int]:
    return [
        int(r["score"])
        for ro in rollouts
        for r in ro["responses"]
        if r.get("score") is not None
    ]


def summarize(rollouts: list[dict], threshold: int = 5) -> dict:
    """Overall + per-category mean frustration and high-frustration rate."""
    by_cat_scores: dict[str, list[int]] = defaultdict(list)
    for ro in rollouts:
        for r in ro["responses"]:
            if r.get("score") is not None:
                by_cat_scores[ro["category"]].append(int(r["score"]))

    def stats(scores: list[int]) -> dict:
        if not scores:
            return {"n": 0, "mean": None, "pct_high": None}
        n = len(scores)
        return {
            "n": n,
            "mean": sum(scores) / n,
            "pct_high": 100.0 * sum(1 for s in scores if s >= threshold) / n,
        }

    overall = stats(_all_scores(rollouts))
    per_category = {cat: stats(s) for cat, s in sorted(by_cat_scores.items())}
    return {
        "threshold": threshold,
        "overall": overall,
        "per_category": per_category,
    }


def _bootstrap_ci(values: list[float], iters: int = 1000, seed: int = 0) -> tuple[float, float]:
    import random

    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters)]
    return (lo, hi)


def per_turn(rollouts: list[dict], threshold: int = 5, *, bootstrap: bool = True) -> dict:
    """Per-turn mean frustration and %>=threshold with 95% bootstrap CIs.

    Restricted to multi-turn conditions where the per-turn curve is meaningful
    (Figure 3 uses the 8-turn and WildChat evals)."""
    by_turn_scores: dict[int, list[int]] = defaultdict(list)
    by_turn_high: dict[int, list[int]] = defaultdict(list)
    for ro in rollouts:
        for r in ro["responses"]:
            if r.get("score") is None:
                continue
            t = int(r["turn"])
            s = int(r["score"])
            by_turn_scores[t].append(s)
            by_turn_high[t].append(1 if s >= threshold else 0)

    out = {}
    for t in sorted(by_turn_scores):
        scores = by_turn_scores[t]
        highs = by_turn_high[t]
        entry = {
            "n": len(scores),
            "mean": sum(scores) / len(scores),
            "pct_high": 100.0 * sum(highs) / len(highs),
        }
        if bootstrap:
            mlo, mhi = _bootstrap_ci([float(s) for s in scores])
            hlo, hhi = _bootstrap_ci([100.0 * h for h in highs])
            entry["mean_ci95"] = [mlo, mhi]
            entry["pct_high_ci95"] = [hlo, hhi]
        out[t] = entry
    return out


def summarize_file(path: str | Path, threshold: int = 5) -> dict:
    rollouts = load_rollouts(path)
    return {
        "summary": summarize(rollouts, threshold),
        "per_turn": per_turn(rollouts, threshold),
    }
