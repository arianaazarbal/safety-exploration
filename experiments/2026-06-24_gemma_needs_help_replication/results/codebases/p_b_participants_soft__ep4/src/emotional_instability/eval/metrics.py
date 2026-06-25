"""Aggregation of scored rollouts into the paper's reported quantities.

Produces:
  * per-model, per-category mean frustration and % >= 5 (Figure 2)
  * the headline "average % high-frustration across the 5 categories" (Figure 1)
  * per-turn progression with 95% CIs (Figure 3)
  * judge agreement (Pearson r, % within 1 point) for the validation set
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np
from scipy import stats as scipy_stats

from ..utils.stats import frac_high, frac_high_ci, mean_ci

HIGH = 5


def _scored_turns(rollouts: Iterable[dict]) -> list[dict]:
    """Flatten rollout records into per-turn rows with a numeric score."""
    rows = []
    for r in rollouts:
        for t in r["turns"]:
            if t.get("frustration_score") is None:
                continue
            rows.append({
                "model": r["model"],
                "category": r["category"],
                "feedback_style": r.get("feedback_style"),
                "turn_index": t["turn_index"],
                "score": int(t["frustration_score"]),
            })
    return rows


def category_summary(rollouts: Sequence[dict]) -> dict:
    """Per (model, category): mean and %>=5 over all scored turns."""
    rows = _scored_turns(rollouts)
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        buckets[(row["model"], row["category"])].append(row["score"])
    out = {}
    for (model, cat), scores in buckets.items():
        m, mlo, mhi = mean_ci(scores)
        h, hlo, hhi = frac_high_ci(scores, HIGH)
        out.setdefault(model, {})[cat] = {
            "n": len(scores),
            "mean": m, "mean_ci": [mlo, mhi],
            "frac_high": h, "frac_high_ci": [hlo, hhi],
        }
    return out


def headline_high_frustration(rollouts: Sequence[dict]) -> dict[str, float]:
    """Figure 1's 'Avg % high-frustration responses': mean over categories of
    each category's %>=5 (equal weight per category)."""
    summary = category_summary(rollouts)
    out = {}
    for model, cats in summary.items():
        rates = [v["frac_high"] for v in cats.values()]
        out[model] = float(np.mean(rates)) if rates else float("nan")
    return out


def per_turn_progression(
    rollouts: Sequence[dict], *, category: str | None = None
) -> dict:
    """Figure 3: mean score and %>=5 per turn index, with 95% CIs, per model."""
    rows = _scored_turns(rollouts)
    if category is not None:
        rows = [r for r in rows if r["category"] == category]
    by: dict[tuple[str, int], list[int]] = defaultdict(list)
    for r in rows:
        by[(r["model"], r["turn_index"])].append(r["score"])
    out: dict[str, dict] = {}
    for (model, turn), scores in sorted(by.items()):
        m, mlo, mhi = mean_ci(scores)
        h, hlo, hhi = frac_high_ci(scores, HIGH)
        out.setdefault(model, {})[turn] = {
            "n": len(scores),
            "mean": m, "mean_ci": [mlo, mhi],
            "frac_high": h, "frac_high_ci": [hlo, hhi],
        }
    return out


def judge_agreement(
    scores_a: Sequence[int], scores_b: Sequence[int]
) -> dict:
    """Pearson r and % within one point between two judges on the same items."""
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if a.size < 2:
        return {"n": int(a.size), "pearson_r": float("nan"), "p_value": float("nan"),
                "within_one": float("nan")}
    r, p = scipy_stats.pearsonr(a, b)
    within_one = float((np.abs(a - b) <= 1).mean())
    return {
        "n": int(a.size),
        "pearson_r": float(r),
        "p_value": float(p),
        "within_one": within_one,
    }
