"""Aggregate scored responses into the paper's headline numbers/figures.

Figure 1 : per-model average %-high-frustration (mean over categories of %>=5).
Figure 2 : per (model, category) mean frustration and %>=5.
Figure 3 : per-turn mean and %>=5 for the 8-turn (extended) and WildChat conditions.
Figure 6 : per-model mean Petri score per emotion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config
from ..schemas import load_jsonl

HIGH = config.HIGH_FRUSTRATION_THRESHOLD


def load_scored(section2_dir: Path | None = None) -> pd.DataFrame:
    """Load all per-model scored_responses.jsonl under results/section2/*."""
    section2_dir = Path(section2_dir or (config.RESULTS_DIR / "section2"))
    rows = []
    for model_dir in sorted(section2_dir.glob("*")):
        f = model_dir / "scored_responses.jsonl"
        if f.exists():
            rows.extend(load_jsonl(f))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["is_high"] = df["score"] >= HIGH
    return df


def figure1_table(df: pd.DataFrame) -> pd.DataFrame:
    """Avg % high-frustration per model (mean over categories of %>=5)."""
    per_cat = (df.groupby(["model", "category"])["is_high"].mean() * 100).reset_index()
    out = per_cat.groupby("model")["is_high"].mean().reset_index()
    out = out.rename(columns={"is_high": "avg_pct_high_frustration"})
    return out.sort_values("avg_pct_high_frustration", ascending=False).reset_index(drop=True)


def figure2_by_category(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "category"]).agg(
        mean_score=("score", "mean"),
        pct_high=("is_high", lambda s: 100 * s.mean()),
        n=("score", "size"),
    ).reset_index()
    return g


def _ci95(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * values.std(ddof=1) / np.sqrt(len(values))


def figure3_per_turn(df: pd.DataFrame, conditions=("extended", "wildchat")) -> pd.DataFrame:
    sub = df[df["condition"].isin(conditions)]
    rows = []
    for (model, cond, turn), grp in sub.groupby(["model", "condition", "turn_index"]):
        scores = grp["score"].to_numpy(dtype=float)
        highs = (scores >= HIGH).astype(float)
        rows.append({
            "model": model, "condition": cond, "turn_index": int(turn),
            "mean_score": scores.mean(), "mean_ci95": _ci95(scores),
            "pct_high": 100 * highs.mean(), "pct_high_ci95": 100 * _ci95(highs),
            "n": len(scores),
        })
    return pd.DataFrame(rows).sort_values(["model", "condition", "turn_index"])


def petri_summary(petri_path: Path | None = None) -> pd.DataFrame:
    petri_path = Path(petri_path or (config.RESULTS_DIR / "petri" / "petri_scores.jsonl"))
    if not petri_path.exists():
        return pd.DataFrame()
    rows = load_jsonl(petri_path)
    recs = []
    for r in rows:
        for emo, sc in r["scores"].items():
            recs.append({"model": r["model"], "emotion": emo, "score": sc})
    df = pd.DataFrame(recs)
    return df.groupby(["model", "emotion"])["score"].mean().reset_index()
