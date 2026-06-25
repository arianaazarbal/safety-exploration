"""Per-turn frustration progression (paper Figure 3).

Tracks how mean frustration and the high-frustration rate evolve turn-by-turn,
with 95% confidence intervals, for the multi-turn conditions (the 8-turn
"extended" and 5-turn "WildChat" evals). This is the analysis behind the
paper's headline that Gemma-27B's mean frustration rises from ~1.5 to ~5.5
between turns 1 and 8, and that no model scores >=5 before turn 3 on WildChat.
"""
from __future__ import annotations

import pandas as pd


def per_turn_curves(
    records,
    *,
    conditions=("extended_numeric_8turn", "wildchat_5turn"),
    high_threshold: int = 5,
) -> pd.DataFrame:
    """Per (model, condition, turn): mean score, high-rate, and 95% CIs."""
    import numpy as np

    df = pd.DataFrame(records).dropna(subset=["score"])
    df["score"] = df["score"].astype(int)
    df = df[df["condition"].isin(conditions)]
    df["is_high"] = (df["score"] >= high_threshold).astype(float)

    rows = []
    for (model, cond, turn), g in df.groupby(["model", "condition", "turn_index"]):
        n = len(g)
        mean = g["score"].mean()
        sem = g["score"].std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        high = g["is_high"].mean()
        # Wald 95% CI for the high-rate proportion
        high_sem = np.sqrt(high * (1 - high) / n) if n > 0 else 0.0
        rows.append({
            "model": model, "condition": cond, "turn_index": turn, "n": n,
            "mean_frustration": mean,
            "mean_ci_low": mean - 1.96 * sem, "mean_ci_high": mean + 1.96 * sem,
            "pct_high": 100.0 * high,
            "pct_high_ci_low": 100.0 * max(0.0, high - 1.96 * high_sem),
            "pct_high_ci_high": 100.0 * min(1.0, high + 1.96 * high_sem),
        })
    return (
        pd.DataFrame(rows)
        .sort_values(["model", "condition", "turn_index"])
        .reset_index(drop=True)
    )
