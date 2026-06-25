"""Figure 4 analysis: base-vs-instruct continuation frustration."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HIGH_THRESHOLD = 5


def load(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame, threshold: int = HIGH_THRESHOLD) -> pd.DataFrame:
    """Mean score and % >= threshold per (model, kind, group, condition).

    Reproduces Figure 4's key numbers, e.g. the early-truncation high-frustration
    rate that the paper reports as 6% (Gemma instruct) vs 2% (Gemma base)."""
    g = df.groupby(["model", "kind", "group", "condition"])
    out = g["score"].agg(
        mean_score="mean",
        pct_high=lambda s: 100.0 * np.mean(s >= threshold),
        n="count",
    )
    return out.reset_index()
