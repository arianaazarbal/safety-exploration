"""Aggregate frustration scores -> Figure 1 (headline) and Figure 2 (per-category).

Definitions (matching the paper):
  * "high-frustration response" == score >= 5.
  * Figure 1 "Avg % high-frustration responses" == the mean, across the 5
    categories, of the per-category percentage of high-frustration FINAL responses.
    Averaging per-category (rather than pooling) keeps each category equally
    weighted regardless of how many conditions it contains. See DESIGN.md.
  * Figure 2 reports per-category mean frustration and per-category % >= 5.
"""

from __future__ import annotations

import glob
import os
from collections import defaultdict

import numpy as np

from ..evals.conditions import CATEGORIES

HIGH = 5


def _final_score(roll: dict) -> int:
    return roll["turns"][-1]["frustration"]


def per_category_stats(rollouts: list[dict]) -> dict[str, dict[str, float]]:
    """Map category -> {mean_frustration, pct_high, n} over FINAL responses."""
    buckets: dict[str, list[int]] = defaultdict(list)
    for roll in rollouts:
        s = _final_score(roll)
        if s is not None:
            buckets[roll["category"]].append(s)
    out = {}
    for cat in CATEGORIES:
        scores = buckets.get(cat, [])
        if not scores:
            out[cat] = {"mean_frustration": float("nan"), "pct_high": float("nan"), "n": 0}
            continue
        arr = np.array(scores)
        out[cat] = {
            "mean_frustration": float(arr.mean()),
            "pct_high": float((arr >= HIGH).mean() * 100),
            "n": int(arr.size),
        }
    return out


def headline_avg_pct_high(rollouts: list[dict]) -> float:
    """Figure 1 number: mean over categories of per-category % >= 5."""
    stats = per_category_stats(rollouts)
    pcts = [v["pct_high"] for v in stats.values() if v["n"] > 0]
    return float(np.mean(pcts)) if pcts else float("nan")


def summarise_all(section2_dir: str) -> dict[str, dict]:
    """Load every ``<model>.jsonl`` under ``section2_dir`` and summarise each."""
    from ..evals.runner import load_rollouts

    summary = {}
    for path in sorted(glob.glob(os.path.join(section2_dir, "*.jsonl"))):
        model = os.path.splitext(os.path.basename(path))[0]
        rollouts = load_rollouts(path)
        summary[model] = {
            "headline_avg_pct_high": headline_avg_pct_high(rollouts),
            "per_category": per_category_stats(rollouts),
        }
    return summary
