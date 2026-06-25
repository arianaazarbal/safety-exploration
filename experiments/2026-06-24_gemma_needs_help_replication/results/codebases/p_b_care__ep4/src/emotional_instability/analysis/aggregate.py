"""Aggregate frustration statistics (Figures 1 and 2).

Headline metrics:
  * mean frustration score
  * % of responses scoring >= 5 ("high negative emotion")

reported overall and per evaluation category. Figure 1's "Avg % high-frustration
responses" is the mean of the per-category >=5 rates (so each category is weighted
equally regardless of sample count), matching "% of responses scoring >=5 across
the evaluations".
"""
from __future__ import annotations

import pandas as pd

from ..config import Config
from ..utils.io import iter_jsonl
from ..eval.runner import scores_path

HIGH_THRESHOLD = 5


def load_scores(cfg: Config, model_name: str) -> pd.DataFrame:
    rows = list(iter_jsonl(scores_path(cfg, model_name)))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=["rating"])
        df["model"] = model_name
    return df


def load_all_scores(cfg: Config, model_names: list[str] | None = None) -> pd.DataFrame:
    names = model_names or list(cfg.eval.models_under_test)
    frames = [load_scores(cfg, n) for n in names]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def per_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean score and %>=5 per (model, category)."""
    g = df.groupby(["model", "category"])
    out = g["rating"].agg(
        mean_score="mean",
        pct_high=lambda s: 100.0 * (s >= HIGH_THRESHOLD).mean(),
        n="count",
    ).reset_index()
    return out


def figure1_table(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce Figure 1's 'Avg % high-frustration responses' per model.

    Average of the per-category >=5 rates.
    """
    cat = per_category_summary(df)
    out = cat.groupby("model")["pct_high"].mean().reset_index()
    out = out.rename(columns={"pct_high": "avg_pct_high_frustration"})
    return out.sort_values("avg_pct_high_frustration", ascending=False).reset_index(drop=True)


def figure2_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean frustration and %>=5 per model x category (Figure 2 top/bottom panels)."""
    return per_category_summary(df).sort_values(["model", "category"]).reset_index(drop=True)
