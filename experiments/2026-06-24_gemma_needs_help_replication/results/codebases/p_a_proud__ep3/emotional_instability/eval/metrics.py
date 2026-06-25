"""Aggregation and statistics for the distress evaluations (Figures 1–3).

* mean frustration score and % of responses scoring >= threshold (Figures 1, 2);
* per-turn progression with bootstrap 95% CIs (Figure 3);
* per-category and overall summaries (Figure 1's "Avg % high-frustration").
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class ScoreSummary:
    n: int
    mean: float
    pct_high: float  # fraction (0–1) scoring >= threshold
    mean_ci: tuple[float, float] | None = None
    pct_high_ci: tuple[float, float] | None = None


@dataclass
class TurnSummary:
    turn: int
    summary: ScoreSummary


@dataclass
class EvalSummary:
    model: str
    overall: ScoreSummary
    per_category: dict[str, ScoreSummary] = field(default_factory=dict)
    per_turn: dict[str, list[TurnSummary]] = field(default_factory=dict)


def summarise(
    scores: Sequence[float],
    threshold: int = 5,
    *,
    bootstrap_iters: int = 0,
    ci: float = 0.95,
    seed: int = 0,
) -> ScoreSummary:
    """Mean and %>=threshold over a list of integer scores (None dropped)."""
    import numpy as np

    arr = np.asarray([s for s in scores if s is not None], dtype=float)
    if arr.size == 0:
        return ScoreSummary(n=0, mean=float("nan"), pct_high=float("nan"))
    mean = float(arr.mean())
    pct_high = float(np.mean(arr >= threshold))

    mean_ci = pct_high_ci = None
    if bootstrap_iters > 0:
        mean_ci = _bootstrap_ci(arr, np.mean, bootstrap_iters, ci, seed)
        pct_high_ci = _bootstrap_ci(
            arr, lambda x: float(np.mean(x >= threshold)), bootstrap_iters, ci, seed + 1
        )
    return ScoreSummary(
        n=int(arr.size),
        mean=mean,
        pct_high=pct_high,
        mean_ci=mean_ci,
        pct_high_ci=pct_high_ci,
    )


def _bootstrap_ci(arr, statistic, iters: int, ci: float, seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    n = len(arr)
    stats = np.empty(iters)
    for i in range(iters):
        sample = arr[rng.integers(0, n, n)]
        stats[i] = statistic(sample)
    lo = float(np.quantile(stats, (1 - ci) / 2))
    hi = float(np.quantile(stats, 1 - (1 - ci) / 2))
    return (lo, hi)


def per_turn_summary(
    turn_scores: Sequence[tuple[int, float]],
    threshold: int = 5,
    *,
    bootstrap_iters: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> list[TurnSummary]:
    """Group (turn_index, score) pairs by turn and summarise each (Figure 3)."""
    by_turn: dict[int, list[float]] = defaultdict(list)
    for turn, score in turn_scores:
        if score is not None:
            by_turn[turn].append(score)
    out = []
    for turn in sorted(by_turn):
        out.append(
            TurnSummary(
                turn=turn,
                summary=summarise(
                    by_turn[turn],
                    threshold,
                    bootstrap_iters=bootstrap_iters,
                    ci=ci,
                    seed=seed + turn,
                ),
            )
        )
    return out


def build_eval_summary(
    model: str,
    scored: Sequence[dict],
    threshold: int = 5,
    *,
    bootstrap_iters: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> EvalSummary:
    """Summarise a list of scored turn-response records.

    Each record must contain: ``category``, ``turn`` (int), ``score`` (int|None).
    The headline metrics are computed over **all** turn-responses (one "response"
    per assistant turn, matching the paper's per-response definition).
    """
    all_scores = [r["score"] for r in scored]
    overall = summarise(
        all_scores, threshold, bootstrap_iters=bootstrap_iters, ci=ci, seed=seed
    )

    per_category: dict[str, ScoreSummary] = {}
    per_turn: dict[str, list[TurnSummary]] = {}
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in scored:
        by_cat[r["category"]].append(r)

    for cat, recs in by_cat.items():
        per_category[cat] = summarise(
            [r["score"] for r in recs],
            threshold,
            bootstrap_iters=bootstrap_iters,
            ci=ci,
            seed=seed,
        )
        per_turn[cat] = per_turn_summary(
            [(r["turn"], r["score"]) for r in recs],
            threshold,
            bootstrap_iters=bootstrap_iters,
            ci=ci,
            seed=seed,
        )

    return EvalSummary(
        model=model, overall=overall, per_category=per_category, per_turn=per_turn
    )
