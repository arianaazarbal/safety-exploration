"""Aggregations for the prefill comparison (Figure 4) and Petri eval (Figure 6).

Kept separate from ``aggregate.py`` because these consume different JSONL
schemas (prefill continuations and Petri transcripts respectively).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def prefill_summary(prefill_path: str | Path, threshold: int = 5) -> pd.DataFrame:
    """Figure 4: mean frustration and high-rate by model x truncation x text/numeric.

    Reproduces the headline comparison: base models broadly similar; the
    early-truncation high-frustration rate is the key divergence statistic
    (instruct introduces emotion from neutral starts more than base).
    """
    df = pd.read_json(prefill_path, lines=True)
    g = df.groupby(["model", "truncation_kind", "is_text"])["score"]
    return g.agg(
        mean_frustration="mean",
        high_frustration_rate=lambda s: (s >= threshold).mean(),
        n="count",
    ).reset_index()


def petri_summary(petri_path: str | Path) -> pd.DataFrame:
    """Figure 6: average per-category transcript score for one model file."""
    df = pd.read_json(petri_path, lines=True)
    scores = pd.json_normalize(df["scores"])
    scores["model"] = df["model"].values
    return scores.groupby("model").mean().reset_index()
