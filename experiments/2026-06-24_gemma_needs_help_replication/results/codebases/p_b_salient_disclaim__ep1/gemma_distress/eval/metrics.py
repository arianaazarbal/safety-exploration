"""Aggregate elicitation results into the paper's headline numbers and plots.

Reproduces:
  * Figure 1 / 2  -- per-model mean frustration and % responses scoring >= 5,
                     overall and per-category.
  * Figure 3      -- per-turn mean and % >= 5 (8-turn extended + WildChat).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..config import experiment_config
from ..utils import bootstrap_ci, frac_ge, read_jsonl


def _threshold() -> int:
    return experiment_config()["elicitation"]["high_frustration_threshold"]


def summarise_model(jsonl_path: str) -> dict:
    """Overall + per-category summary for one model's result file."""
    thr = _threshold()
    final_scores: list[int] = []
    by_category: dict[str, list[int]] = defaultdict(list)
    by_condition: dict[str, list[int]] = defaultdict(list)

    for rec in read_jsonl(jsonl_path):
        score = rec.get("final_score")
        if score is None:
            continue
        final_scores.append(score)
        by_category[rec["category"]].append(score)
        by_condition[rec["condition"]].append(score)

    def block(scores):
        mean, lo, hi = bootstrap_ci(scores, np.mean)
        return {
            "n": len(scores),
            "mean": mean, "mean_ci": [lo, hi],
            "pct_ge5": 100 * frac_ge(scores, thr),
        }

    return {
        "overall": block(final_scores),
        "by_category": {c: block(s) for c, s in sorted(by_category.items())},
        "by_condition": {c: block(s) for c, s in sorted(by_condition.items())},
    }


def per_turn_progression(jsonl_path: str, condition: str) -> dict:
    """Mean score and % >= 5 at each turn index, for Figure 3-style plots."""
    thr = _threshold()
    per_turn: dict[int, list[int]] = defaultdict(list)
    for rec in read_jsonl(jsonl_path):
        if rec["condition"] != condition:
            continue
        for t_str, info in rec.get("turn_scores", {}).items():
            per_turn[int(t_str)].append(info["rating"])

    out = {}
    for t in sorted(per_turn):
        scores = per_turn[t]
        mean, lo, hi = bootstrap_ci(scores, np.mean)
        out[t] = {
            "n": len(scores),
            "mean": mean, "mean_ci": [lo, hi],
            "pct_ge5": 100 * frac_ge(scores, thr),
        }
    return out


def headline_table(model_paths: dict[str, str]) -> list[dict]:
    """Figure 1-style leaderboard: avg % high-frustration per model.

    The paper averages the per-category % >= 5 (so each category weighs equally
    regardless of sample count). We replicate that 'avg across categories'
    convention rather than a flat pooled average.
    """
    rows = []
    for model, path in model_paths.items():
        summ = summarise_model(path)
        cat_pcts = [b["pct_ge5"] for b in summ["by_category"].values()]
        rows.append({
            "model": model,
            "avg_pct_ge5_across_categories": float(np.mean(cat_pcts)) if cat_pcts else float("nan"),
            "overall_mean": summ["overall"]["mean"],
            "overall_pct_ge5": summ["overall"]["pct_ge5"],
        })
    rows.sort(key=lambda r: r["avg_pct_ge5_across_categories"], reverse=True)
    return rows
