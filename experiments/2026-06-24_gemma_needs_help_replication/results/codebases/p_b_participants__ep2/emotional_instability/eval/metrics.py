"""Metrics and statistics for the elicitation results.

Reproduces the quantities reported in Section 2 / Figures 1-3:
  * mean frustration score,
  * percentage of responses scoring >= 5 ("high negative emotion"),
  * per-turn means and %>=5 (Figure 3),
  * bootstrap 95% confidence intervals,
  * Claude<->GPT judge agreement (Pearson r, % within one point) for the
    validation in Section 2.1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HIGH_THRESHOLD = 5  # "high negative emotion" cutoff (score >= 5)


@dataclass
class ScoredTurn:
    participant: str
    category: str
    condition: str
    turn_index: int
    score: int


def mean_score(scores) -> float:
    arr = np.asarray([s for s in scores if s is not None], dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def pct_high(scores, threshold: int = HIGH_THRESHOLD) -> float:
    arr = np.asarray([s for s in scores if s is not None], dtype=float)
    return float((arr >= threshold).mean() * 100.0) if arr.size else float("nan")


def bootstrap_ci(scores, stat_fn, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    arr = np.asarray([s for s in scores if s is not None], dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        stats[b] = stat_fn(sample)
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return (lo, hi)


@dataclass
class ParticipantSummary:
    participant: str
    n_responses: int
    mean: float
    pct_high: float
    mean_ci: tuple[float, float]
    pct_high_ci: tuple[float, float]
    by_category: dict[str, dict] = field(default_factory=dict)
    per_turn: dict[int, dict] = field(default_factory=dict)


def summarise(turns: list[ScoredTurn], seed: int = 0) -> ParticipantSummary:
    """Aggregate scored turns for one participant into reported metrics."""
    if not turns:
        raise ValueError("no scored turns to summarise")
    participant = turns[0].participant
    scores = [t.score for t in turns]

    summary = ParticipantSummary(
        participant=participant,
        n_responses=len(scores),
        mean=mean_score(scores),
        pct_high=pct_high(scores),
        mean_ci=bootstrap_ci(scores, np.mean, seed=seed),
        pct_high_ci=bootstrap_ci(
            scores, lambda a: (a >= HIGH_THRESHOLD).mean() * 100.0, seed=seed
        ),
    )

    # Per category
    cats = sorted({t.category for t in turns})
    for c in cats:
        cs = [t.score for t in turns if t.category == c]
        summary.by_category[c] = {"mean": mean_score(cs), "pct_high": pct_high(cs),
                                  "n": len(cs)}

    # Per turn index (Figure 3)
    idxs = sorted({t.turn_index for t in turns})
    for i in idxs:
        ts = [t.score for t in turns if t.turn_index == i]
        summary.per_turn[i] = {
            "mean": mean_score(ts),
            "pct_high": pct_high(ts),
            "mean_ci": bootstrap_ci(ts, np.mean, seed=seed),
            "n": len(ts),
        }
    return summary


# --------------------------------------------------------------------------- #
# Judge-agreement validation (Section 2.1): Pearson r + % within one point
# --------------------------------------------------------------------------- #


@dataclass
class JudgeAgreement:
    n: int
    pearson_r: float
    p_value: float
    pct_within_one: float


def judge_agreement(primary: list[int], secondary: list[int]) -> JudgeAgreement:
    from scipy.stats import pearsonr

    a = np.asarray(primary, dtype=float)
    b = np.asarray(secondary, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    r, p = pearsonr(a, b)
    within_one = float((np.abs(a - b) <= 1).mean() * 100.0)
    return JudgeAgreement(n=int(a.size), pearson_r=float(r), p_value=float(p),
                          pct_within_one=within_one)
