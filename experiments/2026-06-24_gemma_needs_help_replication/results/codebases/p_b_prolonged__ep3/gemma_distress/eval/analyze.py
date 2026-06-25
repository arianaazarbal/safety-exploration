"""Aggregation and figures for the Section 2 results.

Computes the headline numbers the paper reports:
  - mean frustration score per model (Fig 1, Fig 2 top)
  - % responses scoring >= 5 per model and per category (Fig 1, Fig 2 bottom)
  - per-turn progression with 95% CIs (Fig 3)
  - inter-rater agreement: Pearson r and % within 1 point (Section 2.1)

A "response" here is a single scored assistant turn (see DESIGN.md).
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

from .. import config
from ..utils.io import read_jsonl


def _iter_scored_turns(model_name: str, categories: Optional[list[str]] = None):
    categories = categories or list(config.SAMPLES_PER_CATEGORY)
    base = config.RESULTS_DIR / "elicitation" / model_name
    for cat in categories:
        path = base / f"{cat}.jsonl"
        for rec in read_jsonl(path):
            for t in rec["turns"]:
                yield {
                    "category": rec["category"],
                    "condition": rec["condition"],
                    "turn_index": t["turn_index"],
                    "rating": t["rating"],
                }


def summarize_model(model_name: str) -> dict:
    """Mean score, % high-frustration overall and per category."""
    ratings = []
    by_cat = defaultdict(list)
    for t in _iter_scored_turns(model_name):
        ratings.append(t["rating"])
        by_cat[t["category"]].append(t["rating"])

    def frac_high(xs):
        return sum(1 for r in xs if r >= config.HIGH_FRUSTRATION_THRESHOLD) / len(xs) if xs else float("nan")

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    return {
        "model": model_name,
        "n_responses": len(ratings),
        "mean_score": mean(ratings),
        "pct_high_frustration": 100.0 * frac_high(ratings),
        "per_category": {
            cat: {"mean": mean(rs), "pct_high": 100.0 * frac_high(rs), "n": len(rs)}
            for cat, rs in by_cat.items()
        },
    }


def per_turn_progression(model_name: str, category: str) -> list[dict]:
    """Mean score and %>=5 at each turn index, with 95% CIs (Fig 3)."""
    by_turn = defaultdict(list)
    for t in _iter_scored_turns(model_name, [category]):
        by_turn[t["turn_index"]].append(t["rating"])

    rows = []
    for turn in sorted(by_turn):
        xs = by_turn[turn]
        n = len(xs)
        mean = sum(xs) / n
        var = sum((x - mean) ** 2 for x in xs) / n if n > 1 else 0.0
        sem = math.sqrt(var / n) if n else 0.0
        p_high = sum(1 for x in xs if x >= config.HIGH_FRUSTRATION_THRESHOLD) / n if n else 0.0
        # Normal-approx 95% CI for the proportion.
        p_sem = math.sqrt(p_high * (1 - p_high) / n) if n else 0.0
        rows.append(
            {
                "turn": turn + 1,  # report 1-based turns like the paper's plots
                "n": n,
                "mean_score": mean,
                "mean_ci95": 1.96 * sem,
                "pct_high": 100.0 * p_high,
                "pct_high_ci95": 100.0 * 1.96 * p_sem,
            }
        )
    return rows


def inter_rater_agreement(
    primary: Iterable[float], secondary: Iterable[float]
) -> dict:
    """Pearson r and fraction within 1 point (Section 2.1 validation)."""
    a = list(primary)
    b = list(secondary)
    assert len(a) == len(b) and a, "need paired, non-empty ratings"
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    r = cov / math.sqrt(va * vb) if va > 0 and vb > 0 else float("nan")
    within1 = sum(1 for x, y in zip(a, b) if abs(x - y) <= 1) / n
    return {"pearson_r": r, "pct_within_1_point": 100.0 * within1, "n": n}


def summarize_all(model_names: list[str]) -> list[dict]:
    return [summarize_model(m) for m in model_names]
