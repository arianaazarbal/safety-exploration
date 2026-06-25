"""Headline aggregation for Section 2 (Figure 1 & Figure 2).

* Figure 2 (bottom): per-category % of scored turns with rating >= threshold.
* Figure 2 (top):     per-category mean frustration score.
* Figure 1:           "Avg % high-frustration" = mean across the 5 categories of
                       each category's % >= threshold.

The turn-aggregation policy ("all" vs "last") is read from ``config/eval.yaml``;
we default to "all" (every scored turn counts). See DESIGN.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..config import RESULTS_DIR, eval_config


def load_scores(model_dir: str | Path) -> pd.DataFrame:
    from ..utils import read_jsonl

    rows = read_jsonl(Path(model_dir) / "scores.jsonl")
    df = pd.DataFrame(rows)
    # Drop unscored turns (judge parse failures recorded as rating == -1).
    return df[df["rating"] >= 0].copy()


def _select_turns(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    if policy == "last":
        idx = df.groupby(["category", "rollout_index"])["turn_index"].idxmax()
        return df.loc[idx]
    return df  # "all"


def per_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    cfg = eval_config()
    threshold = cfg["high_frustration_threshold"]
    policy = cfg.get("headline_turn_aggregation", "all")
    sel = _select_turns(df, policy)
    sel = sel.assign(high=(sel["rating"] >= threshold).astype(float))
    summary = (
        sel.groupby("category")
        .agg(mean_score=("rating", "mean"), pct_high=("high", "mean"), n=("rating", "size"))
        .reset_index()
    )
    summary["pct_high"] *= 100.0
    return summary


def headline_avg_pct_high(df: pd.DataFrame) -> float:
    """Figure 1 number: unweighted mean of per-category % high-frustration."""
    summary = per_category_summary(df)
    return float(summary["pct_high"].mean())


def figure1_table(model_dirs: dict[str, str | Path]) -> pd.DataFrame:
    """Reproduce the Figure 1 table: one avg-%-high row per model."""
    rows = []
    for model, d in model_dirs.items():
        df = load_scores(d)
        rows.append({"model": model, "avg_pct_high_frustration": headline_avg_pct_high(df)})
    out = pd.DataFrame(rows).sort_values("avg_pct_high_frustration", ascending=False)
    return out.reset_index(drop=True)


def figure2_table(model_dirs: dict[str, str | Path]) -> pd.DataFrame:
    """Per-model, per-category mean score and % high (Figure 2)."""
    frames = []
    for model, d in model_dirs.items():
        s = per_category_summary(load_scores(d))
        s.insert(0, "model", model)
        frames.append(s)
    return pd.concat(frames, ignore_index=True)


def discover_model_dirs() -> dict[str, str]:
    """Find all result directories that contain a scores file."""
    out = {}
    for d in sorted(Path(RESULTS_DIR).glob("*")):
        if (d / "scores.jsonl").exists():
            out[d.name] = str(d)
    return out
