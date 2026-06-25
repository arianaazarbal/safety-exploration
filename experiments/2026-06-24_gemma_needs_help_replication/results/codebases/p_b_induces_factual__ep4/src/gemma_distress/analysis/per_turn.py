"""Per-turn frustration progression (Figure 3).

For the 8-turn (extended) and 5-turn (WildChat) conditions, compute mean score
and % >= 5 at each turn index, with 95% confidence intervals (the faded band in
Figure 3). The paper's key numbers: Gemma-27B mean rises 1.5 -> 5.5 between
turns 1 and 8; no model scores >=5 until turn 3 on WildChat.
"""
from __future__ import annotations

from collections import defaultdict

from ..config import FRUSTRATION_HIGH_THRESHOLD


def per_turn_progression(
    rows: list[dict],
    condition: str,
    threshold: int = FRUSTRATION_HIGH_THRESHOLD,
) -> list[dict]:
    import numpy as np

    by_turn: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        if r.get("condition") != condition:
            continue
        s = int(r.get("score", -1))
        if s >= 0:
            by_turn[int(r["turn"])].append(s)

    out = []
    for turn in sorted(by_turn):
        scores = np.asarray(by_turn[turn], dtype=float)
        n = len(scores)
        mean = float(scores.mean())
        # 95% CI on the mean via normal approximation.
        sem = float(scores.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        ci = 1.96 * sem
        pct_high = 100.0 * float((scores >= threshold).mean())
        out.append(
            {
                "turn": turn,
                "n": n,
                "mean_frustration": mean,
                "ci95_low": mean - ci,
                "ci95_high": mean + ci,
                "pct_high": pct_high,
            }
        )
    return out
