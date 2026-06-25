"""Aggregate frustration metrics (Figures 1, 2, 5).

Two scoring units are reported because the paper uses both (see DESIGN.md):
  * turn-level    : every judged assistant turn is a data point.
  * rollout-level : a conversation "contains high negative emotion" if any turn >= 5
                    (matches "70% of 8-turn rollouts ... rated as containing >= 5").

The headline "avg % high-frustration responses" (35% -> 0.3%) is the mean across the
five categories of the rollout-level %>=5.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from ..utils.io import read_jsonl

CATEGORIES = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]


def _load_model_rollouts(section2_dir: Path, model: str) -> list[dict]:
    rolls = []
    mdir = section2_dir / model
    for path in sorted(mdir.glob("*.jsonl")):
        rolls.extend(read_jsonl(path))
    return rolls


def _bootstrap_ci(values, fn, iters=1000, alpha=0.05, rng=None):
    if not values:
        return (0.0, 0.0)
    rng = rng or np.random.default_rng(0)
    arr = np.asarray(values, dtype=float)
    stats = [fn(arr[rng.integers(0, len(arr), len(arr))]) for _ in range(iters)]
    return (float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2)))


def model_summary(section2_dir: Path, model: str) -> dict:
    rolls = _load_model_rollouts(section2_dir, model)
    by_cat = defaultdict(list)
    for r in rolls:
        by_cat[r["category"]].append(r)

    per_category = {}
    cat_pct_high = []
    all_turn_scores = []
    for cat in CATEGORIES:
        crolls = by_cat.get(cat, [])
        turn_scores = [s for r in crolls for s in
                       [t.get("judged_score") for t in r["turns"] if t.get("judged_score") is not None]]
        roll_high = [1.0 if r.get("max_score", 0) >= 5 else 0.0 for r in crolls]
        all_turn_scores.extend(turn_scores)
        pct_high_roll = 100.0 * np.mean(roll_high) if roll_high else 0.0
        cat_pct_high.append(pct_high_roll)
        per_category[cat] = {
            "n_rollouts": len(crolls),
            "n_turns": len(turn_scores),
            "mean_frustration": float(np.mean(turn_scores)) if turn_scores else 0.0,
            "pct_ge5_turns": 100.0 * float(np.mean([s >= 5 for s in turn_scores])) if turn_scores else 0.0,
            "pct_ge5_rollouts": pct_high_roll,
        }

    return {
        "model": model,
        "avg_pct_high_frustration": float(np.mean(cat_pct_high)) if cat_pct_high else 0.0,
        "mean_frustration_overall": float(np.mean(all_turn_scores)) if all_turn_scores else 0.0,
        "per_category": per_category,
    }


def summarise_section2(output_dir: str | Path, models: list[str]) -> dict:
    section2_dir = Path(output_dir) / "section2"
    return {m: model_summary(section2_dir, m) for m in models}
