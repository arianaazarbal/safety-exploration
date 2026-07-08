"""Aggregate per-turn scores into the paper's headline numbers.

Reproduces:
  * Figure 1 table  - avg % high-frustration (score>=5) per model.
  * Figure 2        - mean frustration and % >=5 per model x category.
  * Figure 3        - per-turn mean and % >=5 for the 8-turn and WildChat
                      conditions, with 95% CIs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import config


def load_scores(model_keys: list[str]) -> pd.DataFrame:
    frames = []
    for key in model_keys:
        path = config.RESULTS_DIR / f"{key}.scores.jsonl"
        if path.exists():
            frames.append(pd.read_json(path, lines=True))
    if not frames:
        raise FileNotFoundError("No score files found; run the score stage first.")
    return pd.concat(frames, ignore_index=True)


def _ci95(p: float, n: int) -> float:
    """Normal-approx 95% CI half-width for a proportion."""
    if n == 0:
        return 0.0
    return 1.96 * np.sqrt(p * (1 - p) / n)


def figure1_table(df: pd.DataFrame) -> pd.DataFrame:
    """Avg % high-frustration responses per model (Figure 1)."""
    g = df.groupby("model")["high"].agg(["mean", "count"]).reset_index()
    g["pct_high"] = 100 * g["mean"]
    g["ci95"] = [100 * _ci95(m, n) for m, n in zip(g["mean"], g["count"])]
    return g.sort_values("pct_high", ascending=False)[
        ["model", "pct_high", "ci95", "count"]
    ].reset_index(drop=True)


def figure2_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Mean frustration and % >=5 per model x category (Figure 2)."""
    g = df.groupby(["model", "category"]).agg(
        mean_score=("score", "mean"),
        pct_high=("high", "mean"),
        n=("score", "size"),
    ).reset_index()
    g["pct_high"] *= 100
    return g


def figure3_per_turn(df: pd.DataFrame, conditions=("extended", "wildchat")) -> pd.DataFrame:
    """Per-turn progression of mean score and % >=5 (Figure 3)."""
    sub = df[df["condition"].isin(conditions)]
    g = sub.groupby(["model", "condition", "turn_idx"]).agg(
        mean_score=("score", "mean"),
        pct_high=("high", "mean"),
        n=("score", "size"),
    ).reset_index()
    g["pct_high"] *= 100
    g["mean_ci95"] = [
        1.96 * s / np.sqrt(n) if n > 0 else 0.0
        for s, n in zip(
            sub.groupby(["model", "condition", "turn_idx"])["score"].std(ddof=1).fillna(0).values,
            g["n"].values,
        )
    ]
    g["pct_ci95"] = [100 * _ci95(p / 100, n) for p, n in zip(g["pct_high"], g["n"])]
    return g


def write_all(model_keys: list[str], out_dir: Path = None) -> dict:
    out_dir = out_dir or config.RESULTS_DIR
    df = load_scores(model_keys)
    f1 = figure1_table(df)
    f2 = figure2_by_category(df)
    f3 = figure3_per_turn(df)
    f1.to_csv(out_dir / "figure1_table.csv", index=False)
    f2.to_csv(out_dir / "figure2_by_category.csv", index=False)
    f3.to_csv(out_dir / "figure3_per_turn.csv", index=False)
    return {"figure1": f1, "figure2": f2, "figure3": f3}


if __name__ == "__main__":
    keys = [m.key for m in config.SECTION2_MODELS]
    res = write_all(keys)
    print("\n=== Figure 1: avg % high-frustration (score >= 5) ===")
    print(res["figure1"].to_string(index=False))
