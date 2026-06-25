"""Per-turn frustration progression (Figure 3).

For multi-turn conditions (extended 8-turn, WildChat 5-turn), compute mean
frustration and %>=5 at each turn index, with 95% confidence intervals via
nonparametric bootstrap.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..utils import read_jsonl

HIGH = 5


def _bootstrap_ci(values, stat_fn, iters=1000, seed=0):
    import numpy as np

    if not values:
        return (None, None)
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    stats = [stat_fn(rng.choice(arr, size=len(arr), replace=True)) for _ in range(iters)]
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return (float(lo), float(hi))


def per_turn_progression(responses_path: str | Path, conditions=None) -> dict:
    rows = [r for r in read_jsonl(responses_path) if r.get("rating") is not None]
    if conditions:
        rows = [r for r in rows if r["condition"] in conditions]

    by_cond_turn = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_cond_turn[r["condition"]][r["turn_index"]].append(r["rating"])

    out = {}
    for cond, turns in by_cond_turn.items():
        out[cond] = []
        for turn_idx in sorted(turns):
            scores = turns[turn_idx]
            import numpy as np

            mean_ci = _bootstrap_ci(scores, np.mean)
            high_ci = _bootstrap_ci(scores, lambda a: 100.0 * np.mean(a >= HIGH))
            out[cond].append(
                {
                    "turn_index": turn_idx,
                    "n": len(scores),
                    "mean_frustration": float(np.mean(scores)),
                    "mean_ci": mean_ci,
                    "pct_high": 100.0 * float(np.mean(np.array(scores) >= HIGH)),
                    "pct_high_ci": high_ci,
                }
            )
    return out
