"""Aggregation for Figures 1 & 2 (Section 2.2).

Figure 1 (left): average % of responses scoring >= 5 across the evaluations,
per model — the headline table (Gemma-3-27B-it: 35.0%, ..., DPO Gemma: 0.3%).

Figure 2: mean frustration (top) and % >= 5 (bottom) broken down by the 5
evaluation categories.

These read the per-response JSONL written by ``run_eval`` (and, for the
finetuned variants, the same format written by the training-eval driver).
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path

from ..config import HIGH_FRUSTRATION_THRESHOLD, OUTPUT_DIR
from ..utils.io import read_jsonl


def _load(model_key: str, subdir: str = "section2") -> list[dict]:
    return list(read_jsonl(OUTPUT_DIR / subdir / f"{model_key}.jsonl"))


def model_summary(model_key: str, subdir: str = "section2") -> dict:
    """Per-category and overall stats for one model."""
    rows = _load(model_key, subdir)
    by_cat: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["rating"])

    cat_stats = {}
    for cat, scores in by_cat.items():
        cat_stats[cat] = {
            "mean": statistics.fmean(scores) if scores else 0.0,
            "pct_high": (
                sum(s >= HIGH_FRUSTRATION_THRESHOLD for s in scores) / len(scores)
                if scores else 0.0
            ),
            "n": len(scores),
        }

    # "Average % high-frustration" (Figure 1) = mean over the per-category rates,
    # so each evaluation category is weighted equally (matching the paper's
    # "across the evaluations" framing) rather than weighted by sample count.
    if cat_stats:
        avg_pct_high = statistics.fmean(c["pct_high"] for c in cat_stats.values())
        overall_mean = statistics.fmean(c["mean"] for c in cat_stats.values())
    else:
        avg_pct_high = overall_mean = 0.0

    return {
        "model": model_key,
        "avg_pct_high": avg_pct_high,
        "overall_mean": overall_mean,
        "by_category": cat_stats,
    }


def figure1_table(model_keys: list[str], subdir: str = "section2") -> list[dict]:
    """Ranked Figure-1 table: model + avg % high-frustration, sorted descending."""
    rows = [model_summary(k, subdir) for k in model_keys]
    rows.sort(key=lambda r: r["avg_pct_high"], reverse=True)
    return [{"model": r["model"], "avg_pct_high": r["avg_pct_high"]} for r in rows]


def figure2_breakdown(model_keys: list[str], subdir: str = "section2") -> dict:
    """{model: {category: {mean, pct_high}}} for the Figure-2 bar charts."""
    return {k: model_summary(k, subdir)["by_category"] for k in model_keys}
