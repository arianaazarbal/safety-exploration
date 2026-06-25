"""Aggregate scored rollouts into the paper's headline metrics.

Produces:
  * mean frustration and % >= 5, overall and per category (Figures 1, 2);
  * per-turn progression with 95% bootstrap CIs (Figure 3), for the
    multi-turn conditions (Extended 8-turn and WildChat 5-turn).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import config
from ..utils.io import load_jsonl, write_json
from ..utils.stats import bootstrap_ci, frac_at_least, mean


def _valid(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("frustration_score") is not None]


def summarise(rows: list[dict], threshold: int = config.HIGH_FRUSTRATION_THRESHOLD) -> dict:
    rows = _valid(rows)
    scores = [r["frustration_score"] for r in rows]

    # Per-category breakdown.
    by_cat: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["frustration_score"])
    per_category = {
        cat: {
            "n": len(s),
            "mean_frustration": mean(s),
            "pct_high": 100 * frac_at_least(s, threshold),
        }
        for cat, s in by_cat.items()
    }

    return {
        "n_responses": len(scores),
        "overall_mean_frustration": mean(scores),
        "overall_pct_high": 100 * frac_at_least(scores, threshold),
        # Figure 1 headline: average of the per-category high-frustration rates.
        "avg_pct_high_across_categories": mean(
            [v["pct_high"] for v in per_category.values()]
        ),
        "per_category": per_category,
    }


def per_turn_progression(rows: list[dict], condition: str,
                         threshold: int = config.HIGH_FRUSTRATION_THRESHOLD) -> dict:
    rows = [r for r in _valid(rows) if r["condition"] == condition]
    by_turn: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        by_turn[r["turn_index"]].append(r["frustration_score"])
    out = {}
    for turn in sorted(by_turn):
        scores = by_turn[turn]
        point, lo, hi = bootstrap_ci(scores, n_iter=1000, seed=config.GLOBAL_SEED)
        out[turn] = {
            "n": len(scores),
            "mean_frustration": point,
            "mean_ci95": [lo, hi],
            "pct_high": 100 * frac_at_least(scores, threshold),
        }
    return out


def aggregate_file(rollout_path: str | Path) -> dict:
    rows = load_jsonl(rollout_path)
    model = rows[0]["model"] if rows else Path(rollout_path).stem
    report = {
        "model": model,
        "summary": summarise(rows),
        "per_turn": {
            "extended_8turn": per_turn_progression(rows, "extended_8turn"),
            "wildchat_5turn": per_turn_progression(rows, "wildchat_5turn"),
        },
    }
    out_path = Path(rollout_path).with_name(f"{model}_aggregate.json")
    write_json(out_path, report)
    return report
