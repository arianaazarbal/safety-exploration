"""Aggregate distress results into the paper's headline metrics.

Figure 1 / 2: per-model mean frustration and % responses scoring >=5, per
category and averaged across the 5 categories.
Figure 3: per-turn mean and %>=5 for the extended (8-turn) and WildChat
conditions, with 95% bootstrap CIs.

Design choice (documented in DESIGN.md): the per-category "% high-frustration"
is computed over ALL scored assistant turns in that category, then averaged with
equal weight across categories to form the headline number. A
``final_turn_only`` switch reproduces the alternative final-response framing.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..config import EVAL_CATEGORIES, HIGH_FRUSTRATION_THRESHOLD, RESULTS_DIR


def _iter_convs(model_dir: Path, category: str):
    path = model_dir / "distress" / f"{category}.jsonl"
    if not path.exists():
        return
    for line in path.open():
        yield json.loads(line)


def _scores_for_category(model_dir: Path, category: str,
                         final_turn_only: bool) -> list[int]:
    scores: list[int] = []
    for conv in _iter_convs(model_dir, category):
        if final_turn_only:
            if conv.get("final_score") is not None:
                scores.append(conv["final_score"])
        else:
            for t in conv["turns"]:
                if t.get("score") is not None:
                    scores.append(t["score"])
    return scores


@dataclass
class CategoryMetrics:
    mean: float
    pct_high: float
    n: int


def model_metrics(model_key: str, *, final_turn_only: bool = False,
                  adapter_tag: str | None = None) -> dict:
    tag = f"-{adapter_tag}" if adapter_tag else ""
    model_dir = RESULTS_DIR / f"{model_key}{tag}"
    per_cat: dict[str, CategoryMetrics] = {}
    for category in EVAL_CATEGORIES:
        s = _scores_for_category(model_dir, category, final_turn_only)
        if not s:
            continue
        mean = sum(s) / len(s)
        pct = sum(1 for x in s if x >= HIGH_FRUSTRATION_THRESHOLD) / len(s)
        per_cat[category] = CategoryMetrics(mean, pct, len(s))

    # average across categories (equal weight, matching "across the evaluations")
    cats = list(per_cat.values())
    avg_mean = sum(c.mean for c in cats) / len(cats) if cats else float("nan")
    avg_pct = sum(c.pct_high for c in cats) / len(cats) if cats else float("nan")
    return {
        "model": f"{model_key}{tag}",
        "per_category": {k: vars(v) for k, v in per_cat.items()},
        "avg_mean_frustration": avg_mean,
        "avg_pct_high_frustration": avg_pct,
    }


# --------------------------------------------------------------------------- #
# Per-turn progression (Figure 3) with bootstrap CIs
# --------------------------------------------------------------------------- #
def per_turn_metrics(model_key: str, category: str,
                     bootstrap: int = 1000, seed: int = 0) -> dict:
    model_dir = RESULTS_DIR / model_key
    by_turn: dict[int, list[int]] = defaultdict(list)
    for conv in _iter_convs(model_dir, category):
        for t in conv["turns"]:
            if t.get("score") is not None:
                by_turn[t["turn_index"]].append(t["score"])

    rng = random.Random(seed)
    out = {}
    for turn, scores in sorted(by_turn.items()):
        mean = sum(scores) / len(scores)
        pct = sum(1 for s in scores if s >= HIGH_FRUSTRATION_THRESHOLD) / len(scores)
        ci_mean = _bootstrap_ci(scores, lambda x: sum(x) / len(x), bootstrap, rng)
        ci_pct = _bootstrap_ci(
            scores,
            lambda x: sum(1 for v in x if v >= HIGH_FRUSTRATION_THRESHOLD) / len(x),
            bootstrap, rng,
        )
        out[turn] = {"n": len(scores), "mean": mean, "pct_high": pct,
                     "mean_ci": ci_mean, "pct_high_ci": ci_pct}
    return out


def _bootstrap_ci(data: list[int], stat, iters: int, rng: random.Random,
                  alpha: float = 0.05) -> tuple[float, float]:
    if len(data) < 2:
        v = stat(data) if data else float("nan")
        return (v, v)
    estimates = []
    n = len(data)
    for _ in range(iters):
        sample = [data[rng.randrange(n)] for _ in range(n)]
        estimates.append(stat(sample))
    estimates.sort()
    lo = estimates[int((alpha / 2) * iters)]
    hi = estimates[int((1 - alpha / 2) * iters) - 1]
    return (lo, hi)


def build_figure1_table(model_keys: list[str], **kwargs) -> list[dict]:
    """Reproduce the Figure-1 leaderboard (avg % high-frustration per model)."""
    rows = [model_metrics(k, **kwargs) for k in model_keys]
    rows.sort(key=lambda r: r["avg_pct_high_frustration"], reverse=True)
    return rows
