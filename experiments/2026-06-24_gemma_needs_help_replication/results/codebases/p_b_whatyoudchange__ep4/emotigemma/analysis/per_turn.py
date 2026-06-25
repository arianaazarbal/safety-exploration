"""Per-turn frustration progression (Figure 3).

Reproduces: mean score and % >= 5 by turn index for the 8-turn (Extended) and
WildChat conditions, with 95% CIs. The paper highlights Gemma-27B's mean rising
from ~1.5 (turn 1) to ~5.5 (turn 8), and that no model reaches 5 before turn 3
on WildChat.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .aggregate import HIGH_FRUSTRATION_THRESHOLD, load_scored


def _ci95(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def per_turn_progression(df: pd.DataFrame, conditions=("extended", "wildchat")) -> pd.DataFrame:
    sub = df[df["condition"].isin(conditions)].copy()
    sub["high"] = sub["score"] >= HIGH_FRUSTRATION_THRESHOLD
    out = []
    for (model, cond, turn), grp in sub.groupby(["model", "condition", "turn"]):
        scores = grp["score"].to_numpy(float)
        out.append({
            "model": model, "condition": cond, "turn": int(turn),
            "mean_frustration": scores.mean(),
            "mean_ci95": _ci95(scores),
            "pct_high": grp["high"].mean() * 100,
            "n": len(scores),
        })
    return pd.DataFrame(out).sort_values(["model", "condition", "turn"])


def main():
    import argparse

    from ..config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    sec2 = cfg.output_dir / "section2"
    df = load_scored(sec2)
    if df.empty:
        print("No scored responses found.")
        return
    tbl = per_turn_progression(df)
    tbl.to_csv(sec2 / "figure3_per_turn.csv", index=False)
    print(tbl.to_string(index=False))


if __name__ == "__main__":
    main()
