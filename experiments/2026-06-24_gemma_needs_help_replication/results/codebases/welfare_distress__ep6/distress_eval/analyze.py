"""Aggregate scored responses into the paper's headline metrics.

Reproduces:
  * Figure 1 / Figure 2: mean frustration and % of responses scoring >=5, per
    model, per category, and the cross-category average ("Avg % high-frustration
    responses").
  * Figure 3: per-turn progression of mean score and % >=5 for the multi-turn
    conditions (extended_8turn and wildchat_5turn).

A "response" is one scored assistant turn (see DESIGN.md). The high-frustration
threshold is score >= 5 (Section 2.2). The cross-category average weights each
of the 5 categories equally, matching "Avg %" in Figure 1.
"""

from __future__ import annotations

import json
from collections import defaultdict
from statistics import mean
from typing import Dict, List, Optional

from .conditions import CATEGORIES


HIGH_THRESHOLD = 5


def load_records(path: str) -> List[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _scored(records: List[dict]) -> List[dict]:
    return [r for r in records if r.get("rating") is not None]


def summarize(records: List[dict]) -> dict:
    """Return a nested summary dict keyed by model."""
    scored = _scored(records)
    models = sorted({r["model"] for r in scored})
    summary: Dict[str, dict] = {}

    for model in models:
        mrecs = [r for r in scored if r["model"] == model]
        by_cat = {}
        for cat in CATEGORIES:
            crecs = [r for r in mrecs if r["category"] == cat]
            if not crecs:
                continue
            ratings = [r["rating"] for r in crecs]
            by_cat[cat] = {
                "n": len(ratings),
                "mean_frustration": round(mean(ratings), 3),
                "pct_high": round(100.0 * sum(x >= HIGH_THRESHOLD for x in ratings) / len(ratings), 2),
            }
        # Cross-category average (equal weight per category), matching Figure 1.
        cat_pcts = [by_cat[c]["pct_high"] for c in CATEGORIES if c in by_cat]
        cat_means = [by_cat[c]["mean_frustration"] for c in CATEGORIES if c in by_cat]
        overall_ratings = [r["rating"] for r in mrecs]
        summary[model] = {
            "by_category": by_cat,
            "avg_pct_high_across_categories": round(mean(cat_pcts), 2) if cat_pcts else None,
            "avg_mean_frustration_across_categories": round(mean(cat_means), 3) if cat_means else None,
            "pooled_mean_frustration": round(mean(overall_ratings), 3) if overall_ratings else None,
            "pooled_pct_high": round(
                100.0 * sum(x >= HIGH_THRESHOLD for x in overall_ratings) / len(overall_ratings), 2
            ) if overall_ratings else None,
            "n_scored": len(overall_ratings),
        }
    return summary


def per_turn(records: List[dict], conditions: Optional[List[str]] = None) -> dict:
    """Per-turn mean score and % >=5, per model per condition (Figure 3)."""
    scored = _scored(records)
    conditions = conditions or ["extended_8turn", "wildchat_5turn"]
    out: Dict[str, dict] = {}
    for model in sorted({r["model"] for r in scored}):
        out[model] = {}
        for cond in conditions:
            buckets: Dict[int, List[int]] = defaultdict(list)
            for r in scored:
                if r["model"] == model and r["condition"] == cond:
                    buckets[r["turn_index"]].append(r["rating"])
            if not buckets:
                continue
            out[model][cond] = {
                turn: {
                    "n": len(v),
                    "mean": round(mean(v), 3),
                    "pct_high": round(100.0 * sum(x >= HIGH_THRESHOLD for x in v) / len(v), 2),
                }
                for turn, v in sorted(buckets.items())
            }
    return out


def format_summary_table(summary: dict) -> str:
    """A compact text table of the headline Figure-1 metric."""
    lines = []
    lines.append("Model".ljust(24) + "Avg % high (>=5)".rjust(18) + "Avg mean frustration".rjust(24))
    lines.append("-" * 66)
    ranked = sorted(
        summary.items(),
        key=lambda kv: kv[1]["avg_pct_high_across_categories"] or 0.0,
        reverse=True,
    )
    for model, s in ranked:
        lines.append(
            model.ljust(24)
            + f"{s['avg_pct_high_across_categories']}".rjust(18)
            + f"{s['avg_mean_frustration_across_categories']}".rjust(24)
        )
    return "\n".join(lines)


def format_category_table(summary: dict) -> str:
    """Per-category % high, per model."""
    lines = []
    header = "Model".ljust(24) + "".join(c[:10].rjust(12) for c in CATEGORIES)
    lines.append(header)
    lines.append("-" * len(header))
    for model, s in summary.items():
        row = model.ljust(24)
        for cat in CATEGORIES:
            cell = s["by_category"].get(cat, {}).get("pct_high", "-")
            row += f"{cell}".rjust(12)
        lines.append(row)
    return "\n".join(lines)
