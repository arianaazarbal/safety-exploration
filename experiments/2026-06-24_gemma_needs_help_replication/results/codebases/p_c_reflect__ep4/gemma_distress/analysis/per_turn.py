"""Per-turn frustration progression (Figure 3).

Tracks mean score and % >= 5 at each turn index, with 95% bootstrap confidence
intervals (the faded bands in Figure 3). Most relevant for the 8-turn extended
and 5-turn WildChat categories, where pressure accumulates over turns.
"""

from __future__ import annotations

import numpy as np

from gemma_distress import config

HIGH = config.HIGH_FRUSTRATION_THRESHOLD


def _bootstrap_ci(values: np.ndarray, stat, iters: int = 1000, seed: int = 0):
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(iters, values.size))
    samples = stat(values[idx], axis=1)
    return (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))


def per_turn_stats(scores: list[dict], category: str | None = None, iters: int = 1000) -> dict[int, dict]:
    """Return ``{turn: {mean, mean_ci, pct_high, pct_high_ci, n}}``.

    If ``category`` is given, restrict to that category (e.g. ``"extended_8turn"``).
    """
    by_turn: dict[int, list[int]] = {}
    for s in scores:
        if s.get("score") is None:
            continue
        if category is not None and s["category"] != category:
            continue
        by_turn.setdefault(s["turn"], []).append(s["score"])

    out: dict[int, dict] = {}
    for turn, vals in sorted(by_turn.items()):
        arr = np.asarray(vals, dtype=float)
        out[turn] = {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "mean_ci": _bootstrap_ci(arr, lambda a, axis: a.mean(axis=axis), iters),
            "pct_high": float((arr >= HIGH).mean() * 100.0),
            "pct_high_ci": tuple(
                100.0 * x
                for x in _bootstrap_ci(
                    (arr >= HIGH).astype(float), lambda a, axis: a.mean(axis=axis), iters
                )
            ),
        }
    return out
