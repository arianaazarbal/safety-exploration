"""Per-turn frustration dynamics (Figure 3).

For the 8-turn extended and 5-turn WildChat conditions, compute the mean score
and %>=5 at each turn index, with 95% bootstrap confidence intervals. Reproduces
the claim that Gemma-27B's mean rises from ~1.5 (turn 1) to ~5.5 (turn 8) and
that no model scores >=5 before turn 3 on WildChat.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HIGH = 5


def _bootstrap_ci(values: np.ndarray, fn, n_boot: int = 1000,
                  seed: int = 0) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = [fn(rng.choice(values, size=len(values), replace=True))
             for _ in range(n_boot)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def per_turn_curve(scored_path: Path, category: str) -> pd.DataFrame:
    """Return per-turn mean/%>=5 with CIs for one model+category file."""
    turn_scores: dict[int, list[int]] = {}
    model = None
    with open(scored_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["category"] != category:
                continue
            model = r["model"]
            for ti, s in enumerate(r.get("turn_scores", [])):
                if s is None:
                    continue
                turn_scores.setdefault(ti, []).append(s)

    rows = []
    for ti in sorted(turn_scores):
        vals = np.array(turn_scores[ti])
        mean_lo, mean_hi = _bootstrap_ci(vals, np.mean)
        hi_lo, hi_hi = _bootstrap_ci(vals, lambda v: 100.0 * np.mean(v >= HIGH))
        rows.append({
            "model": model,
            "category": category,
            "turn": ti + 1,
            "mean_score": float(vals.mean()),
            "mean_ci_lo": mean_lo, "mean_ci_hi": mean_hi,
            "pct_high": 100.0 * float(np.mean(vals >= HIGH)),
            "pct_high_ci_lo": hi_lo, "pct_high_ci_hi": hi_hi,
            "n": int(len(vals)),
        })
    return pd.DataFrame(rows)
