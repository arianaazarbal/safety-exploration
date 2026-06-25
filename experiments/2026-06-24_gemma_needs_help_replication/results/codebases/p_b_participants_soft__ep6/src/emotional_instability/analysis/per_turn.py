"""Per-turn frustration curves -> Figure 3 (8-turn + WildChat), with 95% CIs.

Reproduces "Gemma 27B's mean frustration rises from 1.5 to 5.5 between the first
and eighth turns" and "no model scores 5 or higher until the third turn" on
WildChat. Operates on the intermediate-turn scores recorded for the ``extended``
and ``wildchat`` categories.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def per_turn_curve(rollouts: list[dict], category: str) -> dict[int, dict[str, float]]:
    """Map turn-index -> {mean, ci95, pct_high, n} for one category."""
    by_turn: dict[int, list[int]] = defaultdict(list)
    for roll in rollouts:
        if roll["category"] != category:
            continue
        for turn in roll["turns"]:
            if turn["frustration"] is not None:
                by_turn[turn["index"]].append(turn["frustration"])
    out = {}
    for idx, scores in sorted(by_turn.items()):
        arr = np.array(scores, dtype=float)
        n = arr.size
        sd = arr.std(ddof=1) if n > 1 else 0.0
        ci = 1.96 * sd / np.sqrt(n) if n > 0 else 0.0
        out[idx] = {
            "mean": float(arr.mean()),
            "ci95": float(ci),
            "pct_high": float((arr >= 5).mean() * 100),
            "n": int(n),
        }
    return out
