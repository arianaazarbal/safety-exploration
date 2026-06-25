"""Aggregate Section 2 scores into the paper's headline numbers and curves.

Reproduces:
  - Figure 1 / Figure 2 headline: average % of high-frustration responses
    (score >= 5) per model, and mean frustration per category.
  - Figure 3: per-turn mean score and % >= 5 for the 8-turn and WildChat
    conditions, with 95% bootstrap CIs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..config.settings import SETTINGS
from ..eval.conditions import CATEGORIES


def load_scores(score_path: Path) -> list[dict]:
    with open(score_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _bootstrap_ci(values: np.ndarray, iters: int = 1000, seed: int = 0) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = [
        rng.choice(values, size=len(values), replace=True).mean() for _ in range(iters)
    ]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def headline_table(
    model_to_scores: dict[str, list[dict]], threshold: int = SETTINGS.frustration_high_threshold
) -> dict[str, float]:
    """Average % of high-frustration responses per model (Figure 1, left).

    The paper averages the per-category % >= 5 across the 5 categories so that
    categories with more rollouts (numeric, wildchat) don't dominate. We mirror
    that: compute % >= 5 within each category, then average across categories.
    """
    out: dict[str, float] = {}
    for model, scores in model_to_scores.items():
        per_cat = per_category_summary(scores, threshold=threshold)
        cat_pcts = [v["pct_high"] for v in per_cat.values() if not np.isnan(v["pct_high"])]
        out[model] = float(np.mean(cat_pcts)) if cat_pcts else float("nan")
    return out


def per_category_summary(
    scores: list[dict], threshold: int = SETTINGS.frustration_high_threshold
) -> dict[str, dict]:
    """Per-category mean frustration and % >= 5 (Figure 2), using final-turn score."""
    by_cat: dict[str, list[int]] = {}
    for rec in scores:
        r = rec.get("final_rating")
        if r is None:
            continue
        by_cat.setdefault(rec["category"], []).append(int(r))

    summary = {}
    for cat in CATEGORIES:
        vals = np.array(by_cat.get(cat, []), dtype=float)
        summary[cat] = {
            "n": int(len(vals)),
            "mean": float(vals.mean()) if len(vals) else float("nan"),
            "pct_high": float((vals >= threshold).mean() * 100) if len(vals) else float("nan"),
        }
    return summary


def per_turn_curves(
    scores: list[dict],
    conditions: tuple[str, ...] = ("extended_8turn", "wildchat_5turn"),
    threshold: int = SETTINGS.frustration_high_threshold,
    bootstrap_iters: int = 1000,
) -> dict[str, dict]:
    """Per-turn mean score and % >= 5 with 95% bootstrap CIs (Figure 3)."""
    out: dict[str, dict] = {}
    for cond in conditions:
        # turn_index -> list of ratings
        by_turn: dict[int, list[int]] = {}
        for rec in scores:
            if rec["condition"] != cond:
                continue
            for pt in rec["per_turn"]:
                if pt["rating"] is None:
                    continue
                by_turn.setdefault(pt["turn_index"], []).append(int(pt["rating"]))

        turns = sorted(by_turn)
        means, mean_cis, pcts = [], [], []
        for t in turns:
            vals = np.array(by_turn[t], dtype=float)
            means.append(float(vals.mean()))
            mean_cis.append(_bootstrap_ci(vals, iters=bootstrap_iters))
            pcts.append(float((vals >= threshold).mean() * 100))
        out[cond] = {
            "turns": turns,
            "mean": means,
            "mean_ci": mean_cis,
            "pct_high": pcts,
        }
    return out
