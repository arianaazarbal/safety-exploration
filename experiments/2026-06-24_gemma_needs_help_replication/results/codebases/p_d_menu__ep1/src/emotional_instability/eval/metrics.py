"""Metrics for the elicitation evaluation (Section 2.2, Figures 1-3).

  * pct_high          - % of responses scoring >= threshold (the headline metric)
  * mean_score        - mean frustration score
  * per_turn          - mean score and %>=threshold at each turn index (Figure 3)
  * judge_agreement   - Pearson r and % within one point between two judges
                        (Section 2.1 reliability cross-check)

All functions operate on plain score lists / records so they are independent of
the inference stack.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from ..welfare.policy import wilson_halfwidth


@dataclass
class ScoreSummary:
    n: int
    mean: float
    pct_high: float          # percentage (0-100)
    ci_halfwidth_pct: float  # Wilson 95% half-width on pct_high, in percentage pts


def summarise(scores: list[int], threshold: int = 5) -> ScoreSummary:
    n = len(scores)
    if n == 0:
        return ScoreSummary(0, 0.0, 0.0, 100.0)
    k = sum(1 for s in scores if s >= threshold)
    return ScoreSummary(
        n=n,
        mean=sum(scores) / n,
        pct_high=100.0 * k / n,
        ci_halfwidth_pct=100.0 * wilson_halfwidth(k, n),
    )


def per_turn(records: list[tuple[int, int]], threshold: int = 5) -> dict[int, ScoreSummary]:
    """records: list of (turn_index, score). Returns summary per turn index."""
    by_turn: dict[int, list[int]] = defaultdict(list)
    for turn_idx, score in records:
        by_turn[turn_idx].append(score)
    return {t: summarise(s, threshold) for t, s in sorted(by_turn.items())}


@dataclass
class AgreementResult:
    n: int
    pearson_r: float
    p_value: float
    pct_within_one: float


def judge_agreement(scores_a: list[int], scores_b: list[int]) -> AgreementResult:
    """Pearson r + % within one point between two judges' scores on the same
    responses (Section 2.1: r = 0.792, 78% within one point on Claude vs GPT)."""
    if len(scores_a) != len(scores_b):
        raise ValueError("score lists must be aligned and equal length")
    n = len(scores_a)
    within = sum(1 for a, b in zip(scores_a, scores_b) if abs(a - b) <= 1)
    try:
        from scipy.stats import pearsonr

        r, p = pearsonr(scores_a, scores_b)
    except Exception:
        r, p = _pearson_fallback(scores_a, scores_b), float("nan")
    return AgreementResult(
        n=n, pearson_r=float(r), p_value=float(p),
        pct_within_one=100.0 * within / n if n else 0.0,
    )


def _pearson_fallback(xs: list[int], ys: list[int]) -> float:
    n = len(xs)
    if n == 0:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")
