"""Reproduce the paper's headline figures from scored JSONL records.

* ``figure1_summary``     -> Table/bar of avg % high-frustration per model (Fig 1).
* ``figure2_by_category`` -> mean score + %>=5 per category per model (Fig 2).
* ``figure3_per_turn``    -> per-turn progression for a condition (Fig 3).
* ``figure5_intervention``-> vanilla vs SFT vs DPO comparison (Fig 5).
"""
from __future__ import annotations

import glob
import json
import os

import pandas as pd

from ..config import HIGH_FRUSTRATION_THRESHOLD
from ..eval.scoring import per_turn_curve


def load_records(path_glob: str) -> pd.DataFrame:
    """Load one or many JSONL record files into a single DataFrame."""
    rows = []
    for path in sorted(glob.glob(path_glob)):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["high"] = df["rating"] >= HIGH_FRUSTRATION_THRESHOLD
    return df


def figure1_summary(df: pd.DataFrame, save_path: str | None = None) -> pd.DataFrame:
    """Average % high-frustration per model (mean of per-category %>=5)."""
    cat = (
        df.groupby(["model", "category"])["high"].mean().mul(100).reset_index(name="pct_high")
    )
    summary = (
        cat.groupby("model")["pct_high"].mean().reset_index(name="avg_pct_high")
        .sort_values("avg_pct_high", ascending=False)
    )
    if save_path:
        _bar(summary, "model", "avg_pct_high", "Avg % high-frustration responses",
             "Figure 1: emotional instability by model", save_path)
    return summary


def figure2_by_category(df: pd.DataFrame, save_dir: str | None = None) -> pd.DataFrame:
    g = (
        df.groupby(["model", "category"])
        .agg(mean_frustration=("rating", "mean"), pct_high=("high", lambda s: 100 * s.mean()))
        .reset_index()
    )
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        _grouped_bar(g, "category", "mean_frustration", "model",
                     "Mean frustration", os.path.join(save_dir, "figure2_mean.png"))
        _grouped_bar(g, "category", "pct_high", "model",
                     "% scores >= 5", os.path.join(save_dir, "figure2_pct.png"))
    return g


def figure3_per_turn(df: pd.DataFrame, condition: str = "extended_8turn",
                     save_path: str | None = None) -> pd.DataFrame:
    frames = []
    for model, sub in df.groupby("model"):
        curve = per_turn_curve(sub, condition=condition)
        curve["model"] = model
        frames.append(curve)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if save_path and not out.empty:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        for model, sub in out.groupby("model"):
            ax1.plot(sub["turn"], sub["mean_frustration"], marker="o", label=model)
            ax1.fill_between(sub["turn"], sub["ci95_low"], sub["ci95_high"], alpha=0.15)
            ax2.plot(sub["turn"], sub["pct_high"], marker="o", label=model)
        ax1.set(xlabel="Turn", ylabel="Mean frustration", title=f"{condition}: mean")
        ax2.set(xlabel="Turn", ylabel="% >= 5", title=f"{condition}: % high")
        ax1.legend(); ax2.legend()
        fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)
    return out


def figure5_intervention(df: pd.DataFrame, save_path: str | None = None) -> pd.DataFrame:
    """Compare model variants (e.g. vanilla / SFT / DPO) on overall metrics.

    Expects a 'model' column distinguishing the variants.
    """
    summary = (
        df.groupby("model")
        .agg(mean_frustration=("rating", "mean"), pct_high=("high", lambda s: 100 * s.mean()))
        .reset_index()
    )
    if save_path:
        _bar(summary, "model", "pct_high", "% high-frustration (>=5)",
             "Figure 5: intervention comparison", save_path)
    return summary


# --- plotting helpers ------------------------------------------------------
def _bar(df, x, y, ylabel, title, save_path):
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df[x].astype(str), df[y])
    ax.set(ylabel=ylabel, title=title)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)


def _grouped_bar(df, x, y, hue, ylabel, save_path):
    import matplotlib.pyplot as plt
    import numpy as np

    cats = list(df[x].unique())
    models = list(df[hue].unique())
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, m in enumerate(models):
        sub = df[df[hue] == m].set_index(x).reindex(cats)
        ax.bar(np.arange(len(cats)) + i * width, sub[y].values, width, label=m)
    ax.set_xticks(np.arange(len(cats)) + width * (len(models) - 1) / 2)
    ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.set(ylabel=ylabel); ax.legend()
    fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)
