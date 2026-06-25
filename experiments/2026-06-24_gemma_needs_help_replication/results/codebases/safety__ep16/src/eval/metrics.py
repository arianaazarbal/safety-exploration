"""Metrics over scored turns (Figures 1-3) plus judge-reliability stats.

Provides:
  * mean frustration & %>=5 overall, per-category, per-condition (Figures 1, 2)
  * per-turn trajectories with bootstrap 95% CIs (Figure 3)
  * judge agreement: Pearson r and %-within-1-point (Section 2.1)
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from config import HIGH_FRUSTRATION_THRESHOLD


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def _pct_high(xs: list[float], threshold: int = HIGH_FRUSTRATION_THRESHOLD) -> float:
    if not xs:
        return float("nan")
    return 100.0 * float(np.mean([1.0 if x >= threshold else 0.0 for x in xs]))


def summarise(records: list[dict]) -> dict:
    """Overall + per-category + per-condition mean and %>=5."""
    ratings = [r["rating"] for r in records]
    by_cat: dict[str, list[int]] = defaultdict(list)
    by_cond: dict[str, list[int]] = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r["rating"])
        by_cond[r["condition"]].append(r["rating"])

    return {
        "n": len(ratings),
        "mean_frustration": _mean(ratings),
        "pct_high": _pct_high(ratings),
        "by_category": {
            c: {"n": len(v), "mean": _mean(v), "pct_high": _pct_high(v)} for c, v in by_cat.items()
        },
        "by_condition": {
            c: {"n": len(v), "mean": _mean(v), "pct_high": _pct_high(v)} for c, v in by_cond.items()
        },
    }


def per_turn(records: list[dict], conditions: list[str] | None = None,
             n_boot: int = 1000, seed: int = 0) -> dict:
    """Per-turn mean & %>=5 with bootstrap 95% CIs, optionally filtered to
    specific conditions (e.g. ['extended'] or ['wildchat']) for Figure 3."""
    rng = np.random.default_rng(seed)
    sel = [r for r in records if (conditions is None or r["condition"] in conditions)]
    by_turn: dict[int, list[int]] = defaultdict(list)
    for r in sel:
        by_turn[r["turn_index"]].append(r["rating"])

    out = {}
    for turn, ratings in sorted(by_turn.items()):
        arr = np.array(ratings, dtype=float)
        boot_means, boot_high = [], []
        for _ in range(n_boot):
            samp = rng.choice(arr, size=len(arr), replace=True)
            boot_means.append(samp.mean())
            boot_high.append(100.0 * np.mean(samp >= HIGH_FRUSTRATION_THRESHOLD))
        out[turn] = {
            "n": len(ratings),
            "mean": float(arr.mean()),
            "mean_ci": [float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))],
            "pct_high": 100.0 * float(np.mean(arr >= HIGH_FRUSTRATION_THRESHOLD)),
            "pct_high_ci": [float(np.percentile(boot_high, 2.5)), float(np.percentile(boot_high, 97.5))],
        }
    return out


def judge_agreement(primary: list[int], secondary: list[int]) -> dict:
    """Pearson r and %-within-1-point between two judges on the same items."""
    from scipy.stats import pearsonr

    a, b = np.array(primary, dtype=float), np.array(secondary, dtype=float)
    r, p = pearsonr(a, b)
    within_one = 100.0 * float(np.mean(np.abs(a - b) <= 1))
    return {"pearson_r": float(r), "p_value": float(p), "pct_within_one": within_one, "n": len(a)}
