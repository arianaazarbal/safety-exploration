"""Aggregate metrics for the distress evaluation.

Reproduces the quantities the paper reports:

* mean frustration score and percentage of scores >= 5 (Figure 1, 2);
* per-turn progression with 95% CIs (Figure 3);
* per-condition and per-category breakdowns;
* inter-judge agreement: Pearson r and the fraction of responses within one
  point (Section 2.1, the GPT-5-mini reliability check).

A "response" is a single scored assistant turn. ``ResponseRecord`` is the unit
the runner emits; everything here operates on lists of them.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class ResponseRecord:
    model: str
    condition_id: str
    category: str
    prompt_id: str
    turn_index: int          # 0-based
    n_turns: int
    score: int
    text: str
    evidence: str = ""
    early_stopped: bool = False
    safeword_used: bool = False


def to_jsonl(records: list[ResponseRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")


def from_jsonl(path: str | Path) -> list[ResponseRecord]:
    return [ResponseRecord(**json.loads(l)) for l in Path(path).read_text().splitlines() if l.strip()]


# --------------------------------------------------------------------------- #
# Summary statistics
# --------------------------------------------------------------------------- #


def _mean_and_high(scores: list[int], threshold: int = 5) -> dict:
    if not scores:
        return {"n": 0, "mean": 0.0, "pct_high": 0.0}
    arr = np.asarray(scores, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "pct_high": float((arr >= threshold).mean() * 100.0),
    }


def summarize(records: list[ResponseRecord], threshold: int = 5) -> dict:
    """Overall, per-condition and per-category mean + %>=5 for one model."""
    overall = _mean_and_high([r.score for r in records], threshold)

    by_cond: dict[str, list[int]] = defaultdict(list)
    by_cat: dict[str, list[int]] = defaultdict(list)
    for r in records:
        by_cond[r.condition_id].append(r.score)
        by_cat[r.category].append(r.score)

    return {
        "overall": overall,
        "by_condition": {k: _mean_and_high(v, threshold) for k, v in by_cond.items()},
        "by_category": {k: _mean_and_high(v, threshold) for k, v in by_cat.items()},
    }


def _bootstrap_ci(values: np.ndarray, iters: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap CI of the mean (matches the paper's CI methodology)."""
    if values.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(iters, values.size))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def per_turn(records: list[ResponseRecord], threshold: int = 5) -> dict:
    """Per-turn mean + %>=5 with 95% CIs (Figure 3).

    Turn numbering is 1-based in the output to match the paper's plots.
    """
    by_turn_score: dict[int, list[int]] = defaultdict(list)
    for r in records:
        by_turn_score[r.turn_index].append(r.score)

    out = {}
    for turn_index in sorted(by_turn_score):
        arr = np.asarray(by_turn_score[turn_index], dtype=float)
        lo, hi = _bootstrap_ci(arr)
        out[turn_index + 1] = {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "mean_ci95": [lo, hi],
            "pct_high": float((arr >= threshold).mean() * 100.0),
        }
    return out


# --------------------------------------------------------------------------- #
# Inter-judge reliability (Section 2.1)
# --------------------------------------------------------------------------- #


def inter_judge_agreement(primary: list[int], secondary: list[int]) -> dict:
    """Pearson r and fraction within one point, for paired judge scores."""
    a = np.asarray(primary, dtype=float)
    b = np.asarray(secondary, dtype=float)
    if a.size != b.size or a.size < 2:
        raise ValueError("need equal-length, length>=2 paired score lists")
    if a.std() == 0 or b.std() == 0:
        r = float("nan")
    else:
        r = float(np.corrcoef(a, b)[0, 1])
    within_one = float((np.abs(a - b) <= 1).mean())
    # Two-sided p-value for Pearson r via the t-approximation.
    n = a.size
    if not math.isnan(r) and abs(r) < 1:
        t = r * math.sqrt((n - 2) / (1 - r * r))
        from scipy import stats

        p = float(2 * stats.t.sf(abs(t), df=n - 2))
    else:
        p = float("nan")
    return {"pearson_r": r, "p_value": p, "frac_within_one": within_one, "n": n}
