"""Per-turn frustration progression (Figure 3).

For the multi-turn settings (the 8-turn Extended and 5-turn WildChat
conditions), the paper plots mean frustration and % >= 5 as a function of turn
index, with 95% CI bands. Gemma-3-27B's mean rises from ~1.5 (turn 1) to ~5.5
(turn 8); with WildChat prompts, no model scores >= 5 until the third turn.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..utils.stats import mean_and_ci95


def per_turn_progression(
    scored_path: str | Path,
    condition: str,
    threshold: int = 5,
) -> pd.DataFrame:
    """Mean score (+95% CI) and high-frustration rate by turn, for one condition."""
    df = pd.read_json(scored_path, lines=True)
    df = df[df["condition"] == condition]
    rows = []
    for (model, turn), grp in df.groupby(["model", "turn"]):
        scores = grp["score"].tolist()
        mean, lo, hi = mean_and_ci95(scores)
        rows.append(
            {
                "model": model,
                "turn": int(turn),
                "mean_frustration": mean,
                "ci_lo": lo,
                "ci_hi": hi,
                "high_frustration_rate": float((grp["score"] >= threshold).mean()),
                "n": len(scores),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "turn"]).reset_index(drop=True)
