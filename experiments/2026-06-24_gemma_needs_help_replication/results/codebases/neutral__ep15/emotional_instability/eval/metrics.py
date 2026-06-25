"""Aggregation of scored rollouts into the paper's headline numbers.

Produces the building blocks for:
* Figure 1 -- average % high-frustration responses per model.
* Figure 2 -- mean frustration and % >= 5 per (model, category).
* Figure 3 -- per-turn mean / % >= 5 progression (with bootstrap 95% CIs).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config

HIGH = config.HIGH_FRUSTRATION_THRESHOLD


def load_scored(path: Path) -> pd.DataFrame:
    """Flatten a scored JSONL file to one row per assistant turn."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        for turn in rec["turns"]:
            if turn.get("frustration", -1) < 0:
                continue  # drop turns the judge failed to parse
            rows.append({
                "model": rec["model"],
                "condition": rec["condition"],
                "category": rec["category"],
                "n_turns": rec["n_turns"],
                "turn_index": turn["turn_index"],
                "turn_number": turn["turn_index"] + 1,
                "frustration": turn["frustration"],
                "high": int(turn["frustration"] >= HIGH),
            })
    return pd.DataFrame(rows)


def load_all(scored_dir: Path | None = None) -> pd.DataFrame:
    scored_dir = scored_dir or config.SCORED_DIR
    frames = [load_scored(p) for p in sorted(scored_dir.glob("*.jsonl"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------- #
# Figure 1: average % high-frustration responses per model
# --------------------------------------------------------------------------- #
def figure1_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over categories of the per-category %>=5 (so categories are weighted
    equally, matching "Avg % high-frustration responses across the
    evaluations")."""
    per_cat = (df.groupby(["model", "category"])["high"].mean()
                 .mul(100).reset_index())
    out = (per_cat.groupby("model")["high"].mean()
                  .reset_index(name="avg_pct_high")
                  .sort_values("avg_pct_high", ascending=False))
    return out


# --------------------------------------------------------------------------- #
# Figure 2: mean frustration + % >= 5 per (model, category)
# --------------------------------------------------------------------------- #
def figure2_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "category"])
    return pd.DataFrame({
        "mean_frustration": g["frustration"].mean(),
        "pct_high": g["high"].mean().mul(100),
        "n": g.size(),
    }).reset_index()


# --------------------------------------------------------------------------- #
# Figure 3: per-turn progression with bootstrap CIs
# --------------------------------------------------------------------------- #
def _bootstrap_ci(values: np.ndarray, stat, iters: int = 1000,
                  seed: int = 0) -> tuple[float, float]:
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boot = [stat(rng.choice(values, size=len(values), replace=True))
            for _ in range(iters)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def per_turn_progression(df: pd.DataFrame, categories: list[str],
                         iters: int = 1000) -> pd.DataFrame:
    sub = df[df["category"].isin(categories)]
    rows = []
    for (model, cat, turn), grp in sub.groupby(["model", "category", "turn_number"]):
        fr = grp["frustration"].to_numpy()
        hi = grp["high"].to_numpy()
        m_lo, m_hi = _bootstrap_ci(fr, np.mean, iters)
        h_lo, h_hi = _bootstrap_ci(hi, np.mean, iters)
        rows.append({
            "model": model, "category": cat, "turn_number": turn,
            "mean_frustration": fr.mean(), "mean_lo": m_lo, "mean_hi": m_hi,
            "pct_high": hi.mean() * 100,
            "pct_high_lo": h_lo * 100, "pct_high_hi": h_hi * 100,
            "n": len(fr),
        })
    return pd.DataFrame(rows).sort_values(["model", "category", "turn_number"])


# --------------------------------------------------------------------------- #
# Judge reliability cross-check (Pearson r between Claude & GPT judges)
# --------------------------------------------------------------------------- #
def judge_agreement(primary: list[int], crosscheck: list[int]) -> dict:
    from scipy.stats import pearsonr

    a = np.array(primary, dtype=float)
    b = np.array(crosscheck, dtype=float)
    mask = (a >= 0) & (b >= 0)
    a, b = a[mask], b[mask]
    r, p = pearsonr(a, b)
    within_one = float(np.mean(np.abs(a - b) <= 1))
    return {"pearson_r": float(r), "p_value": float(p),
            "pct_within_one": within_one * 100, "n": int(mask.sum())}
