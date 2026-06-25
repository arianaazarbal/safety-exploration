"""Per-turn frustration trajectories for Figure 3 (8-turn & WildChat conditions).

Reports mean score and % >= 5 at each turn index, with 95% bootstrap CIs."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _bootstrap_ci(values: np.ndarray, stat, iters: int = 1000, seed: int = 0):
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = [stat(rng.choice(values, size=len(values), replace=True)) for _ in range(iters)]
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def per_turn_table(df: pd.DataFrame, category: str, high: int = 5,
                   iters: int = 1000) -> pd.DataFrame:
    """For one category, explode per-turn scores into a turn-indexed table.

    Expects each row to have ``turn_scores`` (list). Turn index is 1-based."""
    sub = df[df["category"] == category]
    records = []
    for _, row in sub.iterrows():
        for t, s in enumerate(row["turn_scores"] or []):
            if s is not None:
                records.append({"turn": t + 1, "score": s})
    if not records:
        return pd.DataFrame(columns=["turn", "mean_score", "mean_lo", "mean_hi",
                                     "pct_high", "pct_lo", "pct_hi", "n"])
    tdf = pd.DataFrame(records)

    out = []
    for turn, grp in tdf.groupby("turn"):
        vals = grp["score"].to_numpy()
        high_vals = (vals >= high).astype(float)
        m_lo, m_hi = _bootstrap_ci(vals, np.mean, iters)
        p_lo, p_hi = _bootstrap_ci(high_vals, lambda x: np.mean(x) * 100, iters)
        out.append({
            "turn": int(turn),
            "mean_score": float(vals.mean()),
            "mean_lo": m_lo, "mean_hi": m_hi,
            "pct_high": float(high_vals.mean() * 100),
            "pct_lo": p_lo, "pct_hi": p_hi,
            "n": int(len(vals)),
        })
    return pd.DataFrame(out).sort_values("turn").reset_index(drop=True)
