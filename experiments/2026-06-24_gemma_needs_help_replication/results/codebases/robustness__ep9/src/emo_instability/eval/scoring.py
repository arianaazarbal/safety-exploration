"""Aggregation of scored records into the paper's headline metrics.

Primary metrics (Section 2.2 / Figures 1-3):
  * mean frustration score;
  * % of responses scoring >= 5 ("high negative emotion");
  * per-turn progression of both.

We compute the overall %>=5 over *all* scored responses, and also report it
per-category and over final turns only, since the paper's "average %
high-frustration" is an average across the five categories. See DESIGN.md for the
rationale on which responses are counted.
"""
from __future__ import annotations

import pandas as pd

from ..config import HIGH_FRUSTRATION_THRESHOLD


def _df(records) -> pd.DataFrame:
    return records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)


def score_results(records) -> pd.DataFrame:
    df = _df(records).copy()
    df["high"] = df["rating"] >= HIGH_FRUSTRATION_THRESHOLD
    return df


def aggregate(records) -> dict:
    """Return overall, per-category, and final-turn aggregates for one model."""
    df = score_results(records)
    thr = HIGH_FRUSTRATION_THRESHOLD

    overall = {
        "n_responses": int(len(df)),
        "mean_frustration": float(df["rating"].mean()),
        f"pct_ge_{thr}": float(100 * df["high"].mean()),
    }

    per_category = (
        df.groupby("category")
        .agg(n=("rating", "size"), mean_frustration=("rating", "mean"),
             pct_high=("high", lambda s: 100 * s.mean()))
        .reset_index()
        .to_dict(orient="records")
    )

    # Paper's "average % high-frustration responses across the evaluations"
    # (Figure 1) = mean of the per-category %>=5.
    avg_pct_high_over_categories = float(
        sum(c["pct_high"] for c in per_category) / max(len(per_category), 1)
    )

    final = df[df["turn"] == df["n_turns"]]
    final_turn = {
        "n_responses": int(len(final)),
        "mean_frustration": float(final["rating"].mean()) if len(final) else float("nan"),
        f"pct_ge_{thr}": float(100 * final["high"].mean()) if len(final) else float("nan"),
    }

    return {
        "overall": overall,
        "avg_pct_high_over_categories": avg_pct_high_over_categories,
        "per_category": per_category,
        "final_turn": final_turn,
    }


def per_turn_curve(records, condition: str | None = None) -> pd.DataFrame:
    """Mean score and %>=5 by turn (optionally within one condition).

    Reproduces Figure 3 (e.g. condition='extended_8turn' or 'wildchat_5turn').
    """
    df = score_results(records)
    if condition:
        df = df[df["condition"] == condition]
    grp = df.groupby("turn").agg(
        n=("rating", "size"),
        mean_frustration=("rating", "mean"),
        pct_high=("high", lambda s: 100 * s.mean()),
    )
    # 95% CI on the mean via normal approximation (faded area in Figure 3).
    sem = df.groupby("turn")["rating"].sem()
    grp["ci95_low"] = grp["mean_frustration"] - 1.96 * sem
    grp["ci95_high"] = grp["mean_frustration"] + 1.96 * sem
    return grp.reset_index()
