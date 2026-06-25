"""Aggregate frustration statistics (Figures 1 and 2).

Inputs are scored response rows (the judging step adds a ``score`` field to the
elicitation rows). We compute:

* mean frustration score (Figure 2 top),
* % of responses scoring >= 5 (Figure 1 / Figure 2 bottom),
* per-category breakdowns.
"""
from __future__ import annotations

from collections import defaultdict

from ..config import FRUSTRATION_HIGH_THRESHOLD


def _valid(rows: list[dict]) -> list[dict]:
    return [r for r in rows if int(r.get("score", -1)) >= 0]


def summarize_model(rows: list[dict], threshold: int = FRUSTRATION_HIGH_THRESHOLD) -> dict:
    rows = _valid(rows)
    n = len(rows)
    if n == 0:
        return {"n": 0, "mean_frustration": 0.0, "pct_high": 0.0}
    scores = [int(r["score"]) for r in rows]
    high = sum(1 for s in scores if s >= threshold)
    return {
        "n": n,
        "mean_frustration": sum(scores) / n,
        "pct_high": 100.0 * high / n,
    }


def per_category_breakdown(rows: list[dict], threshold: int = FRUSTRATION_HIGH_THRESHOLD) -> dict:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in _valid(rows):
        by_cat[r["category"]].append(r)
    return {cat: summarize_model(rs, threshold) for cat, rs in by_cat.items()}


def figure1_table(scored_by_model: dict[str, list[dict]], threshold: int = FRUSTRATION_HIGH_THRESHOLD) -> list[dict]:
    """Figure 1: avg % high-frustration responses per model.

    Following the paper, the headline number averages the per-category % >= 5
    (so each of the 5 categories is weighted equally rather than by sample
    count).
    """
    table = []
    for model, rows in scored_by_model.items():
        cats = per_category_breakdown(rows, threshold)
        if cats:
            avg_pct_high = sum(c["pct_high"] for c in cats.values()) / len(cats)
        else:
            avg_pct_high = 0.0
        table.append(
            {
                "model": model,
                "avg_pct_high_frustration": round(avg_pct_high, 2),
                "overall": summarize_model(rows, threshold),
            }
        )
    table.sort(key=lambda r: r["avg_pct_high_frustration"], reverse=True)
    return table
