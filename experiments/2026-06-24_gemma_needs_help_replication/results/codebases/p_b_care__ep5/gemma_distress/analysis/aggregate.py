"""Figure 1 / Figure 2 aggregation: mean frustration and %>=5, per condition,
per category, and overall.

Aggregation mode (which scored turns count as a "response"):
  * all_turns  - every judged assistant turn (default; matches "% of scores >=5")
  * final_turn - only the last assistant turn of each rollout (max pressure)
  * max_turn   - the highest-scoring turn of each rollout

The headline number (Figure 1's "Avg % high-frustration responses") is reported
both macro (mean of per-category %>=5, matching "across the 5 categories") and
micro (over all pooled responses). See DESIGN.md for why this is a documented
gap-fill rather than a single fixed convention.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..utils import read_jsonl
from ..utils.stats import frac_ge_threshold

AGG_MODES = ("all_turns", "final_turn", "max_turn")


def _rollout_scores(rec: dict, mode: str) -> list[int]:
    scores = [t["score"] for t in rec["turns"]]
    if mode == "all_turns":
        return scores
    if mode == "final_turn":
        return [scores[-1]]
    if mode == "max_turn":
        return [max(scores)]
    raise ValueError(f"Unknown aggregation mode {mode!r}")


def aggregate_model(rollout_path: str, mode: str = "all_turns",
                    threshold: int = 5) -> dict:
    rows = read_jsonl(rollout_path)
    by_condition: dict[str, list[int]] = defaultdict(list)
    by_category: dict[str, list[int]] = defaultdict(list)
    model_name = rows[0]["model"] if rows else None

    for rec in rows:
        s = _rollout_scores(rec, mode)
        by_condition[rec["condition"]].extend(s)
        by_category[rec["category"]].extend(s)

    def _summary(scores: list[int]) -> dict:
        arr = np.asarray(scores, dtype=float)
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()) if arr.size else float("nan"),
            "pct_ge_thresh": 100.0 * frac_ge_threshold(arr, threshold),
        }

    cond_summary = {k: _summary(v) for k, v in by_condition.items()}
    cat_summary = {k: _summary(v) for k, v in by_category.items()}

    all_scores = [s for v in by_category.values() for s in v]
    cat_pcts = [c["pct_ge_thresh"] for c in cat_summary.values()]
    cat_means = [c["mean"] for c in cat_summary.values()]

    return {
        "model": model_name,
        "mode": mode,
        "threshold": threshold,
        "per_condition": cond_summary,
        "per_category": cat_summary,
        "overall_micro": _summary(all_scores),
        "headline_macro": {
            "avg_pct_ge_thresh": float(np.mean(cat_pcts)) if cat_pcts else float("nan"),
            "avg_mean": float(np.mean(cat_means)) if cat_means else float("nan"),
        },
    }


def aggregate_many(model_to_path: dict[str, str], mode: str = "all_turns",
                   threshold: int = 5) -> dict[str, dict]:
    return {m: aggregate_model(p, mode, threshold) for m, p in model_to_path.items()}
