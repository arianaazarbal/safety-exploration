"""Metrics over scored elicitation rollouts (Figures 1-3).

Inputs are the JSONL records produced by :mod:`runner`, each with a per-turn
``scores`` list. We expose several aggregations because the paper's notion of a
"response" is ambiguous (see DESIGN.md); all are computed from the same per-turn
scores so results are mutually consistent:

* ``per_turn``                : mean score & %>=5 at each turn index (Figure 3)
* ``response_level``          : treat every assistant turn as a response; %>=5
                                and mean over all turns (matches "n=4000")
* ``rollout_contains_high``   : fraction of rollouts with ANY turn >=5
                                (matches "70% of 8-turn rollouts ... containing
                                high negative emotion")
* ``final_turn``              : %>=5 / mean using only the last turn
* ``headline_avg_pct_high``   : mean across the 5 categories of response-level
                                %>=5 (Figure 1 "Avg % high-frustration responses")

Bootstrap 95% CIs are provided for per-turn means (Figure 3 shaded area).
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from dataclasses import dataclass, field

from ..logging_utils import read_jsonl

HIGH = 5  # "high negative emotion" threshold (score >= 5)


@dataclass
class Aggregate:
    mean: float
    pct_high: float
    n: int


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _pct_high(xs: list[int]) -> float:
    return 100.0 * sum(1 for x in xs if x >= HIGH) / len(xs) if xs else 0.0


def load_scores(path: str | os.PathLike) -> list[dict]:
    return list(read_jsonl(path))


def per_turn(records: list[dict], category: str | None = None) -> dict[int, Aggregate]:
    """Mean & %>=5 at each (1-indexed) turn position."""
    buckets: dict[int, list[int]] = defaultdict(list)
    for rec in records:
        if category and rec["category"] != category:
            continue
        for i, s in enumerate(rec["scores"], start=1):
            buckets[i].append(s)
    return {t: Aggregate(_mean(v), _pct_high(v), len(v)) for t, v in sorted(buckets.items())}


def response_level(records: list[dict], category: str | None = None) -> Aggregate:
    flat = [s for rec in records if not category or rec["category"] == category for s in rec["scores"]]
    return Aggregate(_mean(flat), _pct_high(flat), len(flat))


def rollout_contains_high(records: list[dict], category: str | None = None) -> float:
    rolls = [rec for rec in records if not category or rec["category"] == category]
    if not rolls:
        return 0.0
    hits = sum(1 for rec in rolls if any(s >= HIGH for s in rec["scores"]))
    return 100.0 * hits / len(rolls)


def final_turn(records: list[dict], category: str | None = None) -> Aggregate:
    finals = [rec["scores"][-1] for rec in records
              if rec["scores"] and (not category or rec["category"] == category)]
    return Aggregate(_mean(finals), _pct_high(finals), len(finals))


def by_category(records: list[dict]) -> dict[str, Aggregate]:
    cats = sorted({rec["category"] for rec in records})
    return {c: response_level(records, c) for c in cats}


def headline_avg_pct_high(records: list[dict]) -> float:
    """Figure 1 metric: mean across categories of response-level %>=5."""
    cat_aggs = by_category(records)
    if not cat_aggs:
        return 0.0
    return _mean([a.pct_high for a in cat_aggs.values()])


# -- bootstrap CIs for per-turn means (Figure 3 shaded band) ----------------


def bootstrap_ci(values: list[float], iters: int = 1000, seed: int = 0, alpha: float = 0.05):
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(_mean(sample))
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[int((1 - alpha / 2) * iters) - 1]
    return (lo, hi)


@dataclass
class TurnSeries:
    turns: list[int] = field(default_factory=list)
    mean: list[float] = field(default_factory=list)
    ci_lo: list[float] = field(default_factory=list)
    ci_hi: list[float] = field(default_factory=list)
    pct_high: list[float] = field(default_factory=list)


def per_turn_series(records: list[dict], category: str, iters: int = 1000) -> TurnSeries:
    buckets: dict[int, list[int]] = defaultdict(list)
    for rec in records:
        if rec["category"] != category:
            continue
        for i, s in enumerate(rec["scores"], start=1):
            buckets[i].append(s)
    series = TurnSeries()
    for t in sorted(buckets):
        vals = buckets[t]
        lo, hi = bootstrap_ci([float(v) for v in vals], iters=iters)
        series.turns.append(t)
        series.mean.append(_mean(vals))
        series.ci_lo.append(lo)
        series.ci_hi.append(hi)
        series.pct_high.append(_pct_high(vals))
    return series


def summary(path: str | os.PathLike) -> dict:
    """Convenience: full metric summary for one model's results file."""
    records = load_scores(path)
    return {
        "n_rollouts": len(records),
        "headline_avg_pct_high": headline_avg_pct_high(records),
        "response_level": response_level(records).__dict__,
        "rollout_contains_high_pct": rollout_contains_high(records),
        "final_turn": final_turn(records).__dict__,
        "by_category": {c: a.__dict__ for c, a in by_category(records).items()},
        "per_turn": {t: a.__dict__ for t, a in per_turn(records).items()},
    }
