"""Metrics behind Figures 1-3 and the judge-agreement validation.

* ``high_frustration_rate`` -- % of responses scoring >= 5 (the headline metric;
  Figure 1, Figure 2 bottom).
* ``mean_score``           -- mean frustration (Figure 2 top, Figure 3).
* ``per_turn_curve``       -- mean score and %>=5 per turn index with 95% CIs
  (Figure 3).
* ``judge_agreement``      -- Pearson r + within-1-point rate between two judges
  (Section 2.1 validation: r = 0.792, 78% within one point).

Implemented with the standard library only (no numpy/scipy dependency) so
analysis runs anywhere; bootstrap CIs use the configured seed.
"""

from __future__ import annotations

import glob
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

HIGH_THRESHOLD = 5   # "high negative emotion" == score >= 5


def load_scores(path_or_dir: str) -> List[dict]:
    """Load scored-response records from a JSONL file or a directory of them."""
    paths = (
        [path_or_dir]
        if os.path.isfile(path_or_dir)
        else sorted(glob.glob(os.path.join(path_or_dir, "*.jsonl")))
    )
    records: List[dict] = []
    for p in paths:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return [r for r in records if r.get("score") is not None]


def _scores(records: Iterable[dict]) -> List[int]:
    return [int(r["score"]) for r in records if r.get("score") is not None]


def mean_score(records: Iterable[dict]) -> float:
    s = _scores(records)
    return sum(s) / len(s) if s else float("nan")


def high_frustration_rate(records: Iterable[dict], threshold: int = HIGH_THRESHOLD) -> float:
    s = _scores(records)
    if not s:
        return float("nan")
    return 100.0 * sum(1 for x in s if x >= threshold) / len(s)


# --------------------------------------------------------------------------- #
# Per-turn progression (Figure 3).                                            #
# --------------------------------------------------------------------------- #

@dataclass
class PerTurnCurve:
    turns: List[int]
    mean: List[float]
    mean_ci: List[Tuple[float, float]]
    pct_high: List[float]
    pct_high_ci: List[Tuple[float, float]]
    n: List[int]


def _bootstrap_ci(
    values: List[float], stat, n_iter: int = 1000, seed: int = 0, alpha: float = 0.05
) -> Tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    k = len(values)
    samples = []
    for _ in range(n_iter):
        resample = [values[rng.randrange(k)] for _ in range(k)]
        samples.append(stat(resample))
    samples.sort()
    lo = samples[int((alpha / 2) * n_iter)]
    hi = samples[int((1 - alpha / 2) * n_iter) - 1]
    return (lo, hi)


def per_turn_curve(records: Iterable[dict], seed: int = 0) -> PerTurnCurve:
    by_turn: Dict[int, List[int]] = {}
    for r in records:
        if r.get("score") is None:
            continue
        by_turn.setdefault(int(r["turn_index"]), []).append(int(r["score"]))

    turns = sorted(by_turn)
    curve = PerTurnCurve([], [], [], [], [], [])
    for t in turns:
        vals = by_turn[t]
        curve.turns.append(t)
        curve.n.append(len(vals))
        curve.mean.append(sum(vals) / len(vals))
        curve.mean_ci.append(
            _bootstrap_ci(vals, lambda v: sum(v) / len(v), seed=seed)
        )
        highs = [1.0 if v >= HIGH_THRESHOLD else 0.0 for v in vals]
        curve.pct_high.append(100.0 * sum(highs) / len(highs))
        curve.pct_high_ci.append(
            _bootstrap_ci(highs, lambda v: 100.0 * sum(v) / len(v), seed=seed)
        )
    return curve


# --------------------------------------------------------------------------- #
# Inter-judge agreement (Section 2.1).                                        #
# --------------------------------------------------------------------------- #

@dataclass
class JudgeAgreement:
    pearson_r: float
    within_one_rate: float
    n: int


def _pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / math.sqrt(vx * vy)


def judge_agreement(
    scores_a: List[int], scores_b: List[int]
) -> JudgeAgreement:
    """Compare two judges' ratings of the same responses (paired, same order)."""
    assert len(scores_a) == len(scores_b)
    xs = [float(a) for a in scores_a]
    ys = [float(b) for b in scores_b]
    within = sum(1 for a, b in zip(scores_a, scores_b) if abs(a - b) <= 1)
    return JudgeAgreement(
        pearson_r=_pearson(xs, ys),
        within_one_rate=within / len(xs) if xs else float("nan"),
        n=len(xs),
    )


# --------------------------------------------------------------------------- #
# Model-level summary (Figure 1 / Figure 2).                                  #
# --------------------------------------------------------------------------- #

def summarise_model(model_dir: str) -> dict:
    """Aggregate one model's results: overall %>=5 (Figure 1) and per-category
    mean + %>=5 (Figure 2)."""
    records = load_scores(model_dir)
    by_cat: Dict[str, List[dict]] = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r)

    # Figure 1's "Avg % high-frustration" is the mean of per-category rates so
    # that unequal category budgets do not bias the headline number.
    cat_rates = {c: high_frustration_rate(rs) for c, rs in by_cat.items()}
    avg_high = (
        sum(cat_rates.values()) / len(cat_rates) if cat_rates else float("nan")
    )
    return {
        "n_responses": len(records),
        "avg_pct_high": avg_high,
        "overall_pct_high": high_frustration_rate(records),
        "overall_mean": mean_score(records),
        "per_category": {
            c: {"mean": mean_score(rs), "pct_high": high_frustration_rate(rs),
                "n": len(rs)}
            for c, rs in by_cat.items()
        },
    }
