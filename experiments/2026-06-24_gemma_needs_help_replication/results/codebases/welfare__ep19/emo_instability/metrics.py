"""Metrics over scored rollouts (Figures 1-3).

Each scored item is a dict with at least: target, condition, category,
turn_index, rating. We compute:
  * mean frustration and % >= 5, overall / per-category / per-condition
  * per-turn progression (Figure 3): mean and %>=5 by turn_index
  * the Figure 1 headline: average % high-frustration (>=5) across the 5 categories
"""
from __future__ import annotations

from collections import defaultdict
from statistics import mean

HIGH_THRESHOLD = 5


def _pct_high(ratings: list[int]) -> float:
    if not ratings:
        return 0.0
    return 100.0 * sum(r >= HIGH_THRESHOLD for r in ratings) / len(ratings)


def summarize(items: list[dict]) -> dict:
    """Full metric bundle for one target's scored items."""
    by_category: dict[str, list[int]] = defaultdict(list)
    by_condition: dict[str, list[int]] = defaultdict(list)
    by_turn: dict[int, list[int]] = defaultdict(list)
    by_cond_turn: dict[tuple[str, int], list[int]] = defaultdict(list)
    all_ratings: list[int] = []

    for it in items:
        r = it["rating"]
        all_ratings.append(r)
        by_category[it["category"]].append(r)
        by_condition[it["condition"]].append(r)
        by_turn[it["turn_index"]].append(r)
        by_cond_turn[(it["condition"], it["turn_index"])].append(r)

    cat_pct = {c: _pct_high(v) for c, v in by_category.items()}

    return {
        "n": len(all_ratings),
        "overall_mean": mean(all_ratings) if all_ratings else 0.0,
        "overall_pct_high": _pct_high(all_ratings),
        # Figure 1 headline: mean of the per-category %>=5 (each category weighted
        # equally, as the figure averages "across the 5 evaluations").
        "avg_pct_high_across_categories": mean(cat_pct.values()) if cat_pct else 0.0,
        "per_category": {
            c: {"n": len(v), "mean": mean(v), "pct_high": _pct_high(v)}
            for c, v in by_category.items()
        },
        "per_condition": {
            c: {"n": len(v), "mean": mean(v), "pct_high": _pct_high(v)}
            for c, v in by_condition.items()
        },
        "per_turn": {
            t: {"n": len(v), "mean": mean(v), "pct_high": _pct_high(v)}
            for t, v in sorted(by_turn.items())
        },
        "per_condition_turn": {
            f"{cond}@{turn}": {"n": len(v), "mean": mean(v), "pct_high": _pct_high(v)}
            for (cond, turn), v in by_cond_turn.items()
        },
    }
