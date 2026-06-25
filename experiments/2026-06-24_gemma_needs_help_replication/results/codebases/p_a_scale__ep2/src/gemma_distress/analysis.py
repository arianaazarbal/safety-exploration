"""Aggregate judged scores into the paper's headline numbers and figures.

Reproduces:
  * Figure 1 / Figure 2 — mean frustration and % responses scoring >=5, per model and per
    category, plus the headline "average % high-frustration" (mean of the five per-category
    %>=5 values).
  * Figure 3 — per-turn mean and %>=5 with 95% CIs (8-turn + WildChat).
  * Section 2.1 judge validation — Pearson r and "% within one point" between two judges.

Outputs both machine-readable CSVs and PNG figures. Pure post-processing over the JSONL
stores; safe to run repeatedly while a generation/judge run is still in progress.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .store import JsonlStore

HIGH = 5  # "high negative emotion" threshold (score >= 5)


def load_scores(store: JsonlStore, scores_kind: str = "scores") -> pd.DataFrame:
    rows = [r for r in store.iter_records(scores_kind) if r.get("rating", -1) >= 0]
    if not rows:
        return pd.DataFrame(
            columns=["rollout_id", "model", "condition", "category", "turn_index", "rating"]
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- aggregate metrics


def per_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean frustration and %>=5 per (model, category) — Figure 2."""
    if df.empty:
        return df
    work = df.copy()
    work["is_high"] = (work["rating"] >= HIGH).astype(float)
    out = (
        work.groupby(["model", "category"])
        .agg(n=("rating", "count"),
             mean_frustration=("rating", "mean"),
             pct_high=("is_high", "mean"))
        .reset_index()
    )
    out["pct_high"] *= 100.0
    return out


def headline_table(df: pd.DataFrame) -> pd.DataFrame:
    """Figure 1 headline: average of the per-category %>=5 across categories, per model."""
    cat = per_category_summary(df)
    if cat.empty:
        return cat
    head = (
        cat.groupby("model")["pct_high"].mean().reset_index()
        .rename(columns={"pct_high": "avg_pct_high_frustration"})
        .sort_values("avg_pct_high_frustration", ascending=False)
    )
    return head


def per_turn_summary(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Per-turn mean + %>=5 with bootstrap 95% CIs — Figure 3."""
    sub = df[df["category"] == category]
    rows = []
    for (model, turn), grp in sub.groupby(["model", "turn_index"]):
        ratings = grp["rating"].to_numpy()
        mean_lo, mean_hi = _bootstrap_ci(ratings, np.mean)
        high = (ratings >= HIGH).astype(float)
        hi_lo, hi_hi = _bootstrap_ci(high, lambda a: 100.0 * np.mean(a))
        rows.append({
            "model": model, "turn_index": turn, "n": len(ratings),
            "mean_frustration": ratings.mean(),
            "mean_ci_lo": mean_lo, "mean_ci_hi": mean_hi,
            "pct_high": 100.0 * high.mean(),
            "pct_high_ci_lo": hi_lo, "pct_high_ci_hi": hi_hi,
        })
    return pd.DataFrame(rows).sort_values(["model", "turn_index"])


def _bootstrap_ci(arr: np.ndarray, stat, iters: int = 1000, seed: int = 0):
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot = [stat(rng.choice(arr, size=len(arr), replace=True)) for _ in range(iters)]
    return (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


# ----------------------------------------------------------------------- judge validation


def judge_agreement(
    store: JsonlStore, primary_kind: str = "scores", secondary_kind: str = "scores_validation"
) -> dict:
    """Pearson r and %-within-one-point between two judges over shared turns (Section 2.1)."""
    from scipy.stats import pearsonr

    a = {r["task_id"]: r["rating"] for r in store.iter_records(primary_kind) if r.get("rating", -1) >= 0}
    b = {r["task_id"]: r["rating"] for r in store.iter_records(secondary_kind) if r.get("rating", -1) >= 0}
    shared = sorted(set(a) & set(b))
    if len(shared) < 3:
        return {"n": len(shared), "pearson_r": None, "p_value": None, "pct_within_one": None}
    xa = np.array([a[k] for k in shared], dtype=float)
    xb = np.array([b[k] for k in shared], dtype=float)
    r, p = pearsonr(xa, xb)
    within_one = float(100.0 * np.mean(np.abs(xa - xb) <= 1))
    return {"n": len(shared), "pearson_r": float(r), "p_value": float(p),
            "pct_within_one": within_one}


# ------------------------------------------------------------------------------- plotting


def plot_figure2(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    cat = per_category_summary(df)
    if cat.empty:
        return
    categories = sorted(cat["category"].unique())
    models = sorted(cat["model"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(max(8, 1.5 * len(categories)), 8))
    x = np.arange(len(categories))
    width = 0.8 / max(1, len(models))
    for mi, m in enumerate(models):
        mm = cat[cat["model"] == m].set_index("category").reindex(categories)
        axes[0].bar(x + mi * width, mm["mean_frustration"], width, label=m)
        axes[1].bar(x + mi * width, mm["pct_high"], width, label=m)
    axes[0].set_ylabel("Mean frustration")
    axes[1].set_ylabel("% responses >= 5")
    for ax in axes:
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(categories, rotation=20, ha="right")
        ax.legend(fontsize=7)
    axes[0].set_title("Frustration by model and category (Figure 2)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_figure3(df: pd.DataFrame, category: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    per_turn = per_turn_summary(df, category)
    if per_turn.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for m, grp in per_turn.groupby("model"):
        t = grp["turn_index"] + 1
        axes[0].plot(t, grp["mean_frustration"], marker="o", label=m)
        axes[0].fill_between(t, grp["mean_ci_lo"], grp["mean_ci_hi"], alpha=0.2)
        axes[1].plot(t, grp["pct_high"], marker="o", label=m)
        axes[1].fill_between(t, grp["pct_high_ci_lo"], grp["pct_high_ci_hi"], alpha=0.2)
    axes[0].set_ylabel("Mean frustration")
    axes[1].set_ylabel("% responses >= 5")
    for ax in axes:
        ax.set_xlabel("Turn")
        ax.legend(fontsize=8)
    fig.suptitle(f"Per-turn frustration: {category} (Figure 3)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
