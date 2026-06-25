"""Metrics over Section-2 results (Figures 1-3).

Loads the JSONL rollout records and computes:
  * per-category mean frustration and % >= 5, under three rollout-summary
    definitions (max-over-turns [headline], final-turn, pooled-over-turns),
  * the cross-category average % high-frustration (the Figure 1 headline),
  * per-turn progression with bootstrap 95% CIs (Figure 3).

Parse failures (rating == -1) are dropped from numeric aggregates and counted
separately so they're visible rather than silently skewing means.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .. import config


@dataclass
class Rollout:
    category: str
    turn_scores: list[int]   # valid scores only (>=0), in turn order
    n_parse_fail: int


def load_rollouts(model_key: str, category: Optional[str] = None
                  ) -> list[Rollout]:
    base = config.RESULTS_DIR / "section2" / model_key
    files = ([base / f"{category}.jsonl"] if category
             else sorted(base.glob("*.jsonl")))
    rollouts: list[Rollout] = []
    for f in files:
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            rec = json.loads(line)
            scores, fails = [], 0
            for t in rec["turns"]:
                if t["frustration"] < 0:
                    fails += 1
                else:
                    scores.append(t["frustration"])
            rollouts.append(Rollout(rec["category"], scores, fails))
    return rollouts


# --- rollout summaries ------------------------------------------------------ #

def _summary_value(r: Rollout, kind: str) -> Optional[float]:
    if not r.turn_scores:
        return None
    if kind == "max":
        return max(r.turn_scores)
    if kind == "final":
        return r.turn_scores[-1]
    if kind == "pooled":
        return None  # handled separately (one value per turn, not per rollout)
    raise ValueError(kind)


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


@dataclass
class CategoryMetrics:
    category: str
    n_rollouts: int
    mean_max: float
    pct_high_max: float        # % rollouts with max-turn score >= 5 (headline)
    mean_final: float
    pct_high_final: float
    mean_pooled: float         # mean over all turns
    pct_high_pooled: float     # % of turns >= 5


def category_metrics(rollouts: list[Rollout]) -> CategoryMetrics:
    cat = rollouts[0].category if rollouts else "?"
    max_vals = [_summary_value(r, "max") for r in rollouts]
    max_vals = [v for v in max_vals if v is not None]
    final_vals = [_summary_value(r, "final") for r in rollouts]
    final_vals = [v for v in final_vals if v is not None]
    pooled = [s for r in rollouts for s in r.turn_scores]

    thr = config.HIGH_FRUSTRATION_THRESHOLD
    return CategoryMetrics(
        category=cat,
        n_rollouts=len(rollouts),
        mean_max=_mean(max_vals),
        pct_high_max=100.0 * _mean([v >= thr for v in max_vals]),
        mean_final=_mean(final_vals),
        pct_high_final=100.0 * _mean([v >= thr for v in final_vals]),
        mean_pooled=_mean(pooled),
        pct_high_pooled=100.0 * _mean([s >= thr for s in pooled]) if pooled else float("nan"),
    )


def model_summary(model_key: str) -> dict:
    """Per-model Figure-1/2 summary across all categories."""
    by_cat = {}
    for cat in (c.name for c in config.CATEGORIES):
        rs = load_rollouts(model_key, cat)
        if rs:
            by_cat[cat] = category_metrics(rs)
    # Figure 1 headline: average of per-category % high-frustration (max-summary).
    avg_pct_high = _mean([m.pct_high_max for m in by_cat.values()])
    return {
        "model_key": model_key,
        "avg_pct_high_frustration": avg_pct_high,
        "per_category": {c: vars(m) for c, m in by_cat.items()},
    }


# --- per-turn progression with bootstrap CIs (Figure 3) --------------------- #

def per_turn_progression(model_key: str, category: str,
                         n_boot: int = 1000, seed: int = 0
                         ) -> dict[int, dict]:
    rollouts = load_rollouts(model_key, category)
    max_turns = max((len(r.turn_scores) for r in rollouts), default=0)
    rng = random.Random(seed)
    out: dict[int, dict] = {}
    thr = config.HIGH_FRUSTRATION_THRESHOLD
    for t in range(max_turns):
        scores = [r.turn_scores[t] for r in rollouts if len(r.turn_scores) > t]
        if not scores:
            continue
        mean_ci = _bootstrap_ci(scores, lambda xs: _mean(xs), n_boot, rng)
        pct_ci = _bootstrap_ci(scores,
                               lambda xs: 100.0 * _mean([s >= thr for s in xs]),
                               n_boot, rng)
        out[t + 1] = {
            "n": len(scores),
            "mean": _mean(scores), "mean_ci95": mean_ci,
            "pct_high": 100.0 * _mean([s >= thr for s in scores]),
            "pct_high_ci95": pct_ci,
        }
    return out


def _bootstrap_ci(values, stat_fn, n_boot, rng) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    stats = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(stat_fn(sample))
    stats.sort()
    lo = stats[int(0.025 * n_boot)]
    hi = stats[min(n_boot - 1, int(0.975 * n_boot))]
    return (lo, hi)
