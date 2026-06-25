"""Aggregate prefill-continuation scores into Figure 4 / Figure 8 numbers."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import config

HIGH = config.HIGH_FRUSTRATION_THRESHOLD


def load(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df = df[df["frustration"] >= 0]
    df["high"] = (df["frustration"] >= HIGH).astype(int)
    return df


def figure4_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean frustration + %>=5 per (model, kind, prompt_type, truncation)."""
    g = df.groupby(["model", "kind", "prompt_type", "truncation"])
    return pd.DataFrame({
        "mean_frustration": g["frustration"].mean(),
        "pct_high": g["high"].mean().mul(100),
        "n": g.size(),
    }).reset_index()


def recovery_rate(df: pd.DataFrame) -> pd.DataFrame:
    """% of continuations still scoring >=5 (Figure 8 recovery limitation)."""
    g = df.groupby(["model", "kind"])
    return pd.DataFrame({"pct_high": g["high"].mean().mul(100),
                         "n": g.size()}).reset_index()
