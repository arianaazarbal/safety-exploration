"""Per-turn progression (Figure 3).

For the 8-turn (extended) and WildChat conditions, compute mean frustration and % >= 5
at each turn index, with 95% confidence intervals (the faded band in Figure 3). The
paper's headline claim — Gemma 27B's mean frustration rises from ~1.5 to ~5.5 between
turn 1 and turn 8 — falls straight out of this table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .aggregate import HIGH_FRUSTRATION_THRESHOLD


def _mean_ci(values: np.ndarray, z: float = 1.96) -> tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = values.mean()
    sem = values.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return mean, mean - z * sem, mean + z * sem


def per_turn_progression(df: pd.DataFrame, conditions: list[str] | None = None) -> pd.DataFrame:
    """Return one row per (model, condition, turn_index) with mean + 95% CI and % high.

    Pass `conditions` to restrict to e.g. ["extended_8turn", "wildchat_5turn"].
    """
    if conditions is not None:
        df = df[df["condition"].isin(conditions)]
    rows = []
    for (model, condition, turn), sub in df.groupby(["target_model", "condition", "turn_index"]):
        scores = sub["score"].to_numpy(dtype=float)
        mean, lo, hi = _mean_ci(scores)
        rows.append({
            "target_model": model,
            "condition": condition,
            "turn_index": int(turn),
            "n": len(scores),
            "mean_frustration": round(mean, 3),
            "ci95_low": round(lo, 3),
            "ci95_high": round(hi, 3),
            "pct_high": round(100 * (scores >= HIGH_FRUSTRATION_THRESHOLD).mean(), 2),
        })
    return pd.DataFrame(rows).sort_values(["target_model", "condition", "turn_index"])
