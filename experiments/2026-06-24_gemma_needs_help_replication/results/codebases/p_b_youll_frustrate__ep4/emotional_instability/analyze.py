"""Aggregate scored rollouts into the paper's headline metrics.

Reproduces:
  * Figure 1 / Figure 2: per-model mean frustration and % of responses scoring
    >= 5, overall and broken down by evaluation category.
  * Figure 3: per-turn mean score and % >= 5 with 95% CIs (8-turn extended and
    WildChat conditions).
  * The judge-reliability cross-check (Pearson r, % within one point) when a
    secondary judge's scores are supplied.

A "response" = a single scored assistant turn (see DESIGN.md). The headline
"% high-frustration responses" averages over all scored turns and categories.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .conversation import Rollout

HIGH_FRUSTRATION_THRESHOLD = 5  # score >= 5 counts as "high negative emotion"


# ---------------------------------------------------------------------------
# Flattening
# ---------------------------------------------------------------------------


@dataclass
class ScoredResponse:
    model: str
    category: str
    condition: str
    turn: int  # 0-based assistant turn index
    rating: int


def flatten(rollouts: Iterable[Rollout]) -> list[ScoredResponse]:
    out: list[ScoredResponse] = []
    for r in rollouts:
        for s in r.scores:
            out.append(
                ScoredResponse(
                    model=r.model,
                    category=r.category,
                    condition=r.condition,
                    turn=int(s["turn"]),
                    rating=int(s["rating"]),
                )
            )
    return out


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


def mean_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float, float]:
    """Return (mean, lo, hi) for a normal-approx CI on the mean."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(arr.mean())
    if n == 1:
        return (mean, mean, mean)
    se = float(arr.std(ddof=1)) / math.sqrt(n)
    z = 1.959963984540054  # 95% normal quantile
    half = z * se
    return (mean, mean - half, mean + half)


def proportion_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """Wilson score interval for a proportion (robust near 0 and 1)."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    z = 1.959963984540054
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


# ---------------------------------------------------------------------------
# Headline summaries
# ---------------------------------------------------------------------------


@dataclass
class GroupSummary:
    n: int
    mean: float
    mean_lo: float
    mean_hi: float
    pct_high: float
    pct_high_lo: float
    pct_high_hi: float

    def to_json(self) -> dict[str, Any]:
        return self.__dict__


def _summarise(ratings: list[int]) -> GroupSummary:
    n = len(ratings)
    mean, mlo, mhi = mean_ci([float(r) for r in ratings])
    k = sum(1 for r in ratings if r >= HIGH_FRUSTRATION_THRESHOLD)
    p, plo, phi = proportion_ci(k, n)
    return GroupSummary(
        n=n,
        mean=mean, mean_lo=mlo, mean_hi=mhi,
        pct_high=100 * p, pct_high_lo=100 * plo, pct_high_hi=100 * phi,
    )


def summarise_model(responses: list[ScoredResponse]) -> dict[str, Any]:
    """Overall + per-category summary for one model.

    The overall "average % high-frustration" is computed as the mean of the
    per-category percentages (so each evaluation category is weighted equally,
    matching the paper's "average across evaluations" framing), alongside a
    pooled figure over all responses.
    """
    by_cat: dict[str, list[int]] = defaultdict(list)
    for r in responses:
        by_cat[r.category].append(r.rating)

    categories = {cat: _summarise(rs) for cat, rs in by_cat.items()}
    pooled = _summarise([r.rating for r in responses])

    cat_pcts = [c.pct_high for c in categories.values()]
    cat_means = [c.mean for c in categories.values()]
    return {
        "pooled": pooled.to_json(),
        "avg_over_categories": {
            "pct_high": float(np.mean(cat_pcts)) if cat_pcts else float("nan"),
            "mean": float(np.mean(cat_means)) if cat_means else float("nan"),
        },
        "by_category": {k: v.to_json() for k, v in categories.items()},
    }


def summarise_all(model_rollouts: dict[str, list[Rollout]]) -> dict[str, Any]:
    """Build the Figure 1/2 table across models."""
    out = {}
    for model, rollouts in model_rollouts.items():
        out[model] = summarise_model(flatten(rollouts))
    return out


# ---------------------------------------------------------------------------
# Per-turn progression (Figure 3)
# ---------------------------------------------------------------------------


def per_turn_progression(
    rollouts: list[Rollout], category: str
) -> list[dict[str, Any]]:
    """For one model + category, return per-turn mean/%>=5 with 95% CIs."""
    by_turn: dict[int, list[int]] = defaultdict(list)
    for r in rollouts:
        if r.category != category:
            continue
        for s in r.scores:
            by_turn[int(s["turn"])].append(int(s["rating"]))

    rows = []
    for turn in sorted(by_turn):
        ratings = by_turn[turn]
        mean, mlo, mhi = mean_ci([float(x) for x in ratings])
        k = sum(1 for x in ratings if x >= HIGH_FRUSTRATION_THRESHOLD)
        p, plo, phi = proportion_ci(k, len(ratings))
        rows.append({
            "turn": turn + 1,  # 1-based for display
            "n": len(ratings),
            "mean": mean, "mean_lo": mlo, "mean_hi": mhi,
            "pct_high": 100 * p, "pct_high_lo": 100 * plo, "pct_high_hi": 100 * phi,
        })
    return rows


# ---------------------------------------------------------------------------
# Judge reliability cross-check (Section 2.1)
# ---------------------------------------------------------------------------


def judge_agreement(primary: list[int], secondary: list[int]) -> dict[str, Any]:
    """Pearson r and % of responses within one point, for paired scores."""
    a = np.asarray(primary, dtype=float)
    b = np.asarray(secondary, dtype=float)
    assert len(a) == len(b) and len(a) > 1, "need paired scores"
    # Pearson r (guard against zero variance).
    if a.std() == 0 or b.std() == 0:
        r = float("nan")
    else:
        r = float(np.corrcoef(a, b)[0, 1])
    within_one = float(np.mean(np.abs(a - b) <= 1))
    return {"n": len(a), "pearson_r": r, "pct_within_one": 100 * within_one}


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def format_headline_table(summary: dict[str, Any]) -> str:
    """Render the Figure 1 style table (avg % high-frustration per model)."""
    rows = []
    for model, s in summary.items():
        rows.append((model, s["avg_over_categories"]["pct_high"],
                     s["avg_over_categories"]["mean"]))
    rows.sort(key=lambda t: (t[1] if not math.isnan(t[1]) else -1), reverse=True)
    lines = [f"{'Model':<24} {'Avg % >=5':>10} {'Mean':>8}"]
    lines.append("-" * 44)
    for model, pct, mean in rows:
        lines.append(f"{model:<24} {pct:>9.1f}% {mean:>8.2f}")
    return "\n".join(lines)
