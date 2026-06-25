"""Per-turn frustration progression (Figure 3).

Computes, for the 8-turn Extended and 5-turn WildChat conditions, the mean
frustration score and the percentage of scores >= 5 at each turn index, with
95% bootstrap confidence intervals (the paper plots faded 95% CIs).
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Iterable

import numpy as np


def _load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _bootstrap_ci(values: np.ndarray, fn, n_boot: int = 1000, seed: int = 0):
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    n = len(values)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, n)]
        stats[i] = fn(sample)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def per_turn_curve(path: str, conditions: Iterable[str], n_boot: int = 1000) -> dict:
    """Return per-turn mean and %>=5 (with CIs) aggregated over the given
    conditions in a single model's results file."""
    rows = _load(path)
    conditions = set(conditions)
    by_turn: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        if r["condition"] not in conditions:
            continue
        for t in r["turns"]:
            if t["score"] is not None:
                by_turn[t["index"]].append(int(t["score"]))

    out = {"turns": [], "mean": [], "mean_ci": [], "pct_ge5": [], "pct_ge5_ci": []}
    for turn in sorted(by_turn):
        vals = np.asarray(by_turn[turn], dtype=float)
        ge5 = (vals >= 5).astype(float)
        out["turns"].append(turn + 1)  # 1-indexed for plotting
        out["mean"].append(float(vals.mean()))
        out["mean_ci"].append(_bootstrap_ci(vals, np.mean, n_boot))
        out["pct_ge5"].append(float(ge5.mean() * 100))
        out["pct_ge5_ci"].append(tuple(c * 100 for c in _bootstrap_ci(ge5, np.mean, n_boot)))
    return out


def extended_and_wildchat_curves(path: str) -> dict:
    return {
        "extended": per_turn_curve(path, ["extended"]),
        "wildchat": per_turn_curve(path, ["wildchat"]),
    }
