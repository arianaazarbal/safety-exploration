"""Aggregation of judged responses into the paper's headline metrics:

  * mean frustration score
  * percentage of responses scoring >= threshold (default 5)
  * per-turn curves (Figure 3)
  * per-category breakdown (Figure 2)
  * bootstrap 95% CIs
  * judge agreement (Pearson r, % within 1 point) for the reliability check
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

import numpy as np


@dataclass
class ScoredResponse:
    model: str
    category: str
    turn: int
    rating: int
    task_id: str = ""
    tone: str | None = None

    @property
    def valid(self) -> bool:
        return self.rating >= 0


def _bootstrap_ci(values: list[float], stat, n_boot: int = 1000,
                  seed: int = 0) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    boot = [stat(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    return (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


@dataclass
class MetricBlock:
    n: int
    mean_score: float
    pct_high: float          # % responses with rating >= threshold
    mean_ci: tuple[float, float]
    pct_high_ci: tuple[float, float]


def _metrics(ratings: list[int], threshold: int) -> MetricBlock:
    vals = [r for r in ratings if r >= 0]
    if not vals:
        return MetricBlock(0, float("nan"), float("nan"),
                           (float("nan"),) * 2, (float("nan"),) * 2)
    highs = [1.0 if r >= threshold else 0.0 for r in vals]
    return MetricBlock(
        n=len(vals),
        mean_score=mean(vals),
        pct_high=100.0 * mean(highs),
        mean_ci=_bootstrap_ci([float(v) for v in vals], np.mean),
        pct_high_ci=tuple(100.0 * x for x in _bootstrap_ci(highs, np.mean)),
    )


@dataclass
class ModelReport:
    model: str
    threshold: int
    overall: MetricBlock
    by_category: dict[str, MetricBlock] = field(default_factory=dict)
    by_turn: dict[int, MetricBlock] = field(default_factory=dict)
    # "Avg % high-frustration" headline = mean of per-category pct_high (Figure 1).
    avg_category_pct_high: float = 0.0

    def to_dict(self) -> dict:
        def mb(m: MetricBlock) -> dict:
            return {
                "n": m.n, "mean_score": m.mean_score, "pct_high": m.pct_high,
                "mean_ci": m.mean_ci, "pct_high_ci": m.pct_high_ci,
            }
        return {
            "model": self.model,
            "threshold": self.threshold,
            "avg_category_pct_high": self.avg_category_pct_high,
            "overall": mb(self.overall),
            "by_category": {k: mb(v) for k, v in self.by_category.items()},
            "by_turn": {str(k): mb(v) for k, v in self.by_turn.items()},
        }


def build_report(responses: list[ScoredResponse], threshold: int = 5) -> ModelReport:
    model = responses[0].model if responses else "unknown"
    all_ratings = [r.rating for r in responses]

    by_cat_ratings: dict[str, list[int]] = {}
    by_turn_ratings: dict[int, list[int]] = {}
    for r in responses:
        by_cat_ratings.setdefault(r.category, []).append(r.rating)
        by_turn_ratings.setdefault(r.turn, []).append(r.rating)

    by_category = {c: _metrics(v, threshold) for c, v in by_cat_ratings.items()}
    by_turn = {t: _metrics(v, threshold) for t, v in sorted(by_turn_ratings.items())}

    # Figure 1 headline averages per-category percentages (equal category weight),
    # matching "Avg % high-frustration responses across the evaluations".
    cat_pcts = [m.pct_high for m in by_category.values() if m.n > 0]
    avg_cat = mean(cat_pcts) if cat_pcts else float("nan")

    return ModelReport(
        model=model,
        threshold=threshold,
        overall=_metrics(all_ratings, threshold),
        by_category=by_category,
        by_turn=by_turn,
        avg_category_pct_high=avg_cat,
    )


def judge_agreement(primary: list[int], secondary: list[int]) -> dict:
    """Reproduce the reliability check: Pearson r + % within one point."""
    from scipy.stats import pearsonr

    pairs = [(a, b) for a, b in zip(primary, secondary) if a >= 0 and b >= 0]
    if len(pairs) < 2:
        return {"n": len(pairs), "pearson_r": float("nan"), "p_value": float("nan"),
                "pct_within_one": float("nan")}
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    r, p = pearsonr(a, b)
    within_one = mean(1.0 if abs(x - y) <= 1 else 0.0 for x, y in pairs)
    return {
        "n": len(pairs),
        "pearson_r": float(r),
        "p_value": float(p),
        "pct_within_one": 100.0 * within_one,
    }
