"""Aggregate frustration metrics (Figures 1 and 2).

From a scores JSONL, computes per-condition / per-category / overall:

* mean final-turn frustration score, and
* percentage of final-turn responses scoring >= threshold (default 5, "high negative
  emotion").

The Figure 1 headline ("Avg % high-frustration responses") is the mean across the five
categories of the per-category %>=5, which weights each category equally rather than by
sample count (matching "average across evaluations").
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional

import numpy as np

from ..eval.conditions import CATEGORY_CONDITIONS
from ..utils import load_jsonl

CATEGORIES = list(CATEGORY_CONDITIONS.keys())


def _final_scores(records: Iterable[dict]) -> list[dict]:
    """Keep records with a valid final score."""
    return [r for r in records if r.get("final_score") is not None]


def _summarise(scores: list[int], threshold: int) -> dict:
    if not scores:
        return {"n": 0, "mean": float("nan"), "pct_high": float("nan")}
    arr = np.array(scores, dtype=float)
    return {
        "n": len(scores),
        "mean": float(arr.mean()),
        "pct_high": float((arr >= threshold).mean() * 100.0),
    }


def compute_metrics(scores_jsonl: str, threshold: int = 5) -> dict:
    """Compute per-condition, per-category, and overall metrics for one model's scores.

    Returns a nested dict:
    ``{"model", "by_condition", "by_category", "overall_pooled", "avg_over_categories"}``.
    """
    records = _final_scores(load_jsonl(scores_jsonl))
    model = records[0]["model"] if records else None

    by_condition_scores: dict[str, list[int]] = defaultdict(list)
    by_category_scores: dict[str, list[int]] = defaultdict(list)
    all_scores: list[int] = []
    for r in records:
        s = r["final_score"]
        by_condition_scores[r["condition"]].append(s)
        by_category_scores[r["category"]].append(s)
        all_scores.append(s)

    by_condition = {c: _summarise(v, threshold) for c, v in by_condition_scores.items()}
    by_category = {c: _summarise(v, threshold) for c, v in by_category_scores.items()}

    # Average over the categories that are present (equal weighting per category).
    cat_pcts = [m["pct_high"] for m in by_category.values() if m["n"] > 0]
    cat_means = [m["mean"] for m in by_category.values() if m["n"] > 0]

    return {
        "model": model,
        "threshold": threshold,
        "by_condition": by_condition,
        "by_category": by_category,
        "overall_pooled": _summarise(all_scores, threshold),
        "avg_over_categories": {
            "pct_high": float(np.mean(cat_pcts)) if cat_pcts else float("nan"),
            "mean": float(np.mean(cat_means)) if cat_means else float("nan"),
        },
    }


def compare_models(scores_jsonls: dict[str, str], threshold: int = 5) -> dict:
    """Compute the Figure 1 table: avg %>=5 across categories, per model.

    ``scores_jsonls`` maps model name -> path. Returns ``{model: avg_pct_high}`` sorted
    descending, plus the full per-model metrics.
    """
    per_model = {name: compute_metrics(path, threshold) for name, path in scores_jsonls.items()}
    ranking = sorted(
        ((name, m["avg_over_categories"]["pct_high"]) for name, m in per_model.items()),
        key=lambda kv: (kv[1] if kv[1] == kv[1] else -1),
        reverse=True,
    )
    return {"ranking": ranking, "per_model": per_model}
