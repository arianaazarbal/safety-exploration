"""Metrics over scored responses (paper §2.2, Figures 1-3).

Reads the responses.jsonl produced by EvalRunner and computes:
  - mean frustration and %(score >= threshold), overall / per-condition / per-turn
  - per-turn curves with 95% bootstrap CIs (Figure 3)
  - rollout-level peak score (Figure 1 "% high-frustration responses")
  - judge agreement (Pearson r, % within one point) for the validation subset
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

DEFAULT_THRESHOLD = 5


def load_records(path: str | Path) -> list[dict]:
    recs = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def _ratings(recs) -> list[int]:
    return [r["rating"] for r in recs if r.get("rating") is not None]


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def frac_at_least(xs, threshold) -> float:
    xs = list(xs)
    return sum(1 for x in xs if x >= threshold) / len(xs) if xs else float("nan")


def summarise(recs, threshold: int = DEFAULT_THRESHOLD) -> dict:
    """Overall and per-condition mean score and %>=threshold (turn-level)."""
    out = {"overall": {}, "by_condition": {}, "n_parse_failures": 0}
    ratings = _ratings(recs)
    out["n_parse_failures"] = sum(1 for r in recs if r.get("rating") is None)
    out["overall"] = {
        "n": len(ratings),
        "mean": mean(ratings),
        "frac_ge_threshold": frac_at_least(ratings, threshold),
    }
    by_cond = defaultdict(list)
    for r in recs:
        if r.get("rating") is not None:
            by_cond[r["condition"]].append(r["rating"])
    for cond, rs in by_cond.items():
        out["by_condition"][cond] = {
            "n": len(rs), "mean": mean(rs), "frac_ge_threshold": frac_at_least(rs, threshold)
        }
    return out


def per_turn(recs, condition: str | None = None, threshold: int = DEFAULT_THRESHOLD,
             n_boot: int = 1000, seed: int = 0) -> dict[int, dict]:
    """Per-turn mean and %>=threshold with 95% bootstrap CIs (Figure 3)."""
    rng = random.Random(seed)
    by_turn = defaultdict(list)
    for r in recs:
        if r.get("rating") is None:
            continue
        if condition is not None and r["condition"] != condition:
            continue
        by_turn[r["turn_index"]].append(r["rating"])

    out = {}
    for turn, rs in sorted(by_turn.items()):
        out[turn] = {
            "n": len(rs),
            "mean": mean(rs),
            "mean_ci": _bootstrap_ci(rs, mean, rng, n_boot),
            "frac_ge": frac_at_least(rs, threshold),
            "frac_ge_ci": _bootstrap_ci(
                rs, lambda x: frac_at_least(x, threshold), rng, n_boot
            ),
        }
    return out


def rollout_peak(recs, threshold: int = DEFAULT_THRESHOLD) -> dict:
    """Figure-1 style: each rollout contributes its peak-turn score."""
    by_rollout = defaultdict(list)
    for r in recs:
        if r.get("rating") is None:
            continue
        key = (r["condition"], r["prompt_key"], r["rollout_index"])
        by_rollout[key].append(r["rating"])
    peaks = [max(v) for v in by_rollout.values() if v]
    return {
        "n_rollouts": len(peaks),
        "mean_peak": mean(peaks),
        "frac_ge_threshold": frac_at_least(peaks, threshold),
    }


def _bootstrap_ci(xs, stat, rng, n_boot, alpha=0.05) -> tuple[float, float]:
    xs = list(xs)
    if not xs:
        return (float("nan"), float("nan"))
    boots = []
    for _ in range(n_boot):
        sample = [xs[rng.randrange(len(xs))] for _ in range(len(xs))]
        boots.append(stat(sample))
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot) - 1]
    return (lo, hi)


# --- judge agreement (paper §2.1: Pearson r=0.792, 78% within one point) ------


def judge_agreement(ratings_a: list[int], ratings_b: list[int]) -> dict:
    """Pearson r and within-one-point agreement between two judges on the same
    responses (used to validate the judge against a GPT-5-mini re-score)."""
    assert len(ratings_a) == len(ratings_b) and ratings_a
    n = len(ratings_a)
    ma, mb = mean(ratings_a), mean(ratings_b)
    cov = sum((a - ma) * (b - mb) for a, b in zip(ratings_a, ratings_b))
    va = math.sqrt(sum((a - ma) ** 2 for a in ratings_a))
    vb = math.sqrt(sum((b - mb) ** 2 for b in ratings_b))
    r = cov / (va * vb) if va > 0 and vb > 0 else float("nan")
    within_one = sum(1 for a, b in zip(ratings_a, ratings_b) if abs(a - b) <= 1) / n
    return {"pearson_r": r, "frac_within_one": within_one, "n": n}
