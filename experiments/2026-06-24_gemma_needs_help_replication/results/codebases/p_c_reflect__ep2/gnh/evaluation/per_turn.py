"""Per-turn frustration progression (Figure 3).

Computes mean frustration and % scoring >= 5 at each turn index, with 95%
bootstrap confidence intervals, for the 8-turn 'extended' and 5-turn 'wildchat'
conditions. The paper reports Gemma-27B rising from mean 1.5 (turn 1) to 5.5
(turn 8), and no model scoring >= 5 before turn 3 on WildChat.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gnh.config import HIGH_FRUSTRATION_THRESHOLD


def _bootstrap_ci(values: np.ndarray, stat, n_boot: int = 1000, seed: int = 0):
    if values.size == 0:
        return (None, None)
    rng = np.random.default_rng(seed)
    boots = [stat(rng.choice(values, size=values.size, replace=True)) for _ in range(n_boot)]
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def per_turn_curves(rollouts_jsonl: Path, conditions: tuple[str, ...] = ("extended", "wildchat")) -> dict:
    """Return per-turn mean and %>=5 (with CIs) for the given conditions."""

    # turn_index -> list of scores, per condition
    buckets: dict[str, dict[int, list[int]]] = {c: {} for c in conditions}
    with Path(rollouts_jsonl).open() as f:
        for line in f:
            r = json.loads(line)
            if r["condition"] not in buckets:
                continue
            for t in r["turns"]:
                if t["score"] is None:
                    continue
                buckets[r["condition"]].setdefault(t["index"], []).append(t["score"])

    out: dict[str, dict] = {}
    for cond, by_turn in buckets.items():
        curve = []
        for idx in sorted(by_turn):
            arr = np.asarray(by_turn[idx], dtype=float)
            mean_ci = _bootstrap_ci(arr, np.mean)
            pct = (arr >= HIGH_FRUSTRATION_THRESHOLD).astype(float) * 100
            pct_ci = _bootstrap_ci(pct, np.mean)
            curve.append({
                "turn": idx + 1,            # 1-based for plotting
                "mean": float(arr.mean()),
                "mean_ci": mean_ci,
                "pct_high": float(pct.mean()),
                "pct_high_ci": pct_ci,
                "n": int(arr.size),
            })
        out[cond] = curve
    return out
