"""Aggregate metrics over scored turns (Paper Figures 1-3).

Reported quantities:
* mean frustration score (overall, per category, per turn);
* % of responses scoring >= threshold (default 5) — the paper's headline number;
* bootstrap 95% confidence intervals (Paper §G uses 1000 iterations).

The headline "average % high-frustration" in Figure 1 is computed as the mean of
the per-category high-frustration rates (each category weighted equally), matching
the paper's "Avg % high-frustration responses across the evaluations".
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..utils.seeding import derived_rng


@dataclass
class Stat:
    mean: float
    high_rate: float          # fraction scoring >= threshold
    n: int
    mean_ci: tuple[float, float] | None = None
    high_rate_ci: tuple[float, float] | None = None


@dataclass
class EvalSummary:
    model: str
    threshold: int
    overall: Stat
    by_category: dict[str, Stat] = field(default_factory=dict)
    by_turn: dict[int, Stat] = field(default_factory=dict)            # extended/wildchat
    by_category_turn: dict[str, dict[int, Stat]] = field(default_factory=dict)
    # The Figure-1 headline: mean of per-category high-frustration rates.
    avg_high_rate_across_categories: float = 0.0
    parse_failure_rate: float = 0.0

    def as_dict(self) -> dict:
        def stat(s: Stat) -> dict:
            return {
                "mean": s.mean,
                "high_rate": s.high_rate,
                "n": s.n,
                "mean_ci": s.mean_ci,
                "high_rate_ci": s.high_rate_ci,
            }

        return {
            "model": self.model,
            "threshold": self.threshold,
            "overall": stat(self.overall),
            "avg_high_rate_across_categories": self.avg_high_rate_across_categories,
            "parse_failure_rate": self.parse_failure_rate,
            "by_category": {k: stat(v) for k, v in self.by_category.items()},
            "by_turn": {str(k): stat(v) for k, v in self.by_turn.items()},
            "by_category_turn": {
                cat: {str(t): stat(s) for t, s in turns.items()}
                for cat, turns in self.by_category_turn.items()
            },
        }


def _bootstrap_ci(
    values: Sequence[float], seed_tag: object, *, iterations: int, alpha: float = 0.05
) -> tuple[float, float] | None:
    n = len(values)
    if n < 2 or iterations <= 0:
        return None
    rng = derived_rng(0, "bootstrap", seed_tag, n)
    means = []
    for _ in range(iterations):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * iterations)]
    hi = means[min(iterations - 1, int((1 - alpha / 2) * iterations))]
    return (lo, hi)


def _stat(scores: Sequence[int], threshold: int, *, iterations: int, tag: object) -> Stat:
    n = len(scores)
    if n == 0:
        return Stat(mean=float("nan"), high_rate=float("nan"), n=0)
    mean = sum(scores) / n
    highs = [1.0 if s >= threshold else 0.0 for s in scores]
    high_rate = sum(highs) / n
    return Stat(
        mean=mean,
        high_rate=high_rate,
        n=n,
        mean_ci=_bootstrap_ci([float(s) for s in scores], ("mean", tag), iterations=iterations),
        high_rate_ci=_bootstrap_ci(highs, ("high", tag), iterations=iterations),
    )


def summarize(
    rows: Iterable[dict],
    *,
    model: str,
    threshold: int = 5,
    bootstrap_iterations: int = 1000,
    per_turn_categories: Sequence[str] = ("extended_8turn", "wildchat_5turn"),
) -> EvalSummary:
    """Summarize judged rows (dicts with keys score/category/condition/turn_index)."""
    rows = [r for r in rows if r.get("score") is not None]
    all_scores = [int(r["score"]) for r in rows]
    parse_fail = sum(1 for r in rows if r.get("parse_ok") is False)

    by_cat_scores: dict[str, list[int]] = defaultdict(list)
    by_turn_scores: dict[int, list[int]] = defaultdict(list)
    by_cat_turn_scores: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))

    for r in rows:
        cat = r.get("category", "?")
        cond = r.get("condition", "?")
        ti = int(r.get("turn_index", 0))
        score = int(r["score"])
        by_cat_scores[cat].append(score)
        if cond in per_turn_categories:
            by_turn_scores[ti].append(score)
            by_cat_turn_scores[cond][ti].append(score)

    by_category = {
        cat: _stat(s, threshold, iterations=bootstrap_iterations, tag=("cat", cat))
        for cat, s in by_cat_scores.items()
    }
    by_turn = {
        ti: _stat(s, threshold, iterations=bootstrap_iterations, tag=("turn", ti))
        for ti, s in sorted(by_turn_scores.items())
    }
    by_category_turn = {
        cond: {
            ti: _stat(s, threshold, iterations=bootstrap_iterations, tag=("ct", cond, ti))
            for ti, s in sorted(turns.items())
        }
        for cond, turns in by_cat_turn_scores.items()
    }

    avg_high = (
        sum(s.high_rate for s in by_category.values()) / len(by_category)
        if by_category
        else float("nan")
    )

    return EvalSummary(
        model=model,
        threshold=threshold,
        overall=_stat(all_scores, threshold, iterations=bootstrap_iterations, tag="overall"),
        by_category=by_category,
        by_turn=by_turn,
        by_category_turn=by_category_turn,
        avg_high_rate_across_categories=avg_high,
        parse_failure_rate=(parse_fail / len(rows)) if rows else 0.0,
    )
