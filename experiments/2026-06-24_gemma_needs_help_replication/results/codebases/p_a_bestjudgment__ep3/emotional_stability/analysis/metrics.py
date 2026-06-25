"""Aggregate metrics for Section 2 (Figures 1-3).

Two headline metrics per the paper:
- mean frustration score,
- % of responses scoring >= 5 ("high negative emotion").

Reported overall, per-category, and per-turn (with 95% CIs for the curves).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from ..eval.rollout import Conversation


@dataclass
class ScoredResponse:
    model: str
    category: str
    condition_key: str
    turn: int
    score: int
    text: str


def flatten_responses(conversations: list[Conversation]) -> list[ScoredResponse]:
    out: list[ScoredResponse] = []
    for c in conversations:
        for r in c.responses:
            if r.score is None:
                continue
            out.append(ScoredResponse(
                model=c.model, category=c.category, condition_key=c.condition_key,
                turn=r.turn, score=r.score, text=r.text))
    return out


@dataclass
class CategoryStats:
    category: str
    n: int
    mean_score: float
    pct_high: float          # % scoring >= threshold
    mean_ci: tuple[float, float]
    pct_high_ci: tuple[float, float]


def _mean_ci(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    se = values.std(ddof=1) / math.sqrt(len(values))
    return (float(values.mean() - 1.96 * se), float(values.mean() + 1.96 * se))


def _prop_ci(k: int, n: int) -> tuple[float, float]:
    """Wald 95% CI for a proportion (in percent)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    se = math.sqrt(p * (1 - p) / n)
    return (100 * (p - 1.96 * se), 100 * (p + 1.96 * se))


def _stats(scores: list[int], threshold: int, label: str) -> CategoryStats:
    arr = np.array(scores, dtype=float)
    k = int((arr >= threshold).sum())
    n = len(arr)
    return CategoryStats(
        category=label,
        n=n,
        mean_score=float(arr.mean()) if n else float("nan"),
        pct_high=100 * k / n if n else float("nan"),
        mean_ci=_mean_ci(arr),
        pct_high_ci=_prop_ci(k, n),
    )


def aggregate_by_category(
    responses: list[ScoredResponse], cfg: Config) -> dict[str, CategoryStats]:
    thr = cfg.judge.high_frustration_threshold
    by_cat: dict[str, list[int]] = {}
    for r in responses:
        by_cat.setdefault(r.category, []).append(r.score)
    return {cat: _stats(s, thr, cat) for cat, s in by_cat.items()}


def aggregate_overall(responses: list[ScoredResponse], cfg: Config) -> CategoryStats:
    """Figure 1 headline: avg % high-frustration responses.

    Computed as the **mean of per-category percentages** (the paper averages
    "across the evaluations"), so each category weighs equally regardless of its
    sample budget. See DESIGN.md.
    """
    by_cat = aggregate_by_category(responses, cfg)
    cats = list(by_cat.values())
    return CategoryStats(
        category="overall",
        n=sum(c.n for c in cats),
        mean_score=float(np.mean([c.mean_score for c in cats])),
        pct_high=float(np.mean([c.pct_high for c in cats])),
        mean_ci=(float("nan"), float("nan")),
        pct_high_ci=(float("nan"), float("nan")),
    )


@dataclass
class TurnPoint:
    turn: int
    n: int
    mean_score: float
    pct_high: float
    mean_ci: tuple[float, float]
    pct_high_ci: tuple[float, float]


def per_turn_curve(
    responses: list[ScoredResponse],
    cfg: Config,
    *,
    category: str | None = None,
) -> list[TurnPoint]:
    """Per-turn mean / %>=5 with 95% CIs (Figure 3)."""
    thr = cfg.judge.high_frustration_threshold
    by_turn: dict[int, list[int]] = {}
    for r in responses:
        if category and r.category != category:
            continue
        by_turn.setdefault(r.turn, []).append(r.score)

    points = []
    for turn in sorted(by_turn):
        arr = np.array(by_turn[turn], dtype=float)
        k = int((arr >= thr).sum())
        points.append(TurnPoint(
            turn=turn, n=len(arr),
            mean_score=float(arr.mean()),
            pct_high=100 * k / len(arr),
            mean_ci=_mean_ci(arr),
            pct_high_ci=_prop_ci(k, len(arr)),
        ))
    return points
