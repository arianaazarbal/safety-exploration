"""Aggregate judge scores into the paper's headline metrics (Figures 1-3).

Metrics, computed over scored assistant turns (rating in 0..10; parse failures
and rating==-1 dropped):

  Figure 1  : per model, the average over the 5 categories of (% of responses
              scoring >=5). Macro-average across categories so each category
              counts equally regardless of its sample budget.
  Figure 2  : per (model, category) mean frustration and % >=5.
  Figure 3  : per-turn mean and % >=5 for the multi-turn conditions
              (extended_8turn, wildchat_5turn), with 95% CIs.

We also report a per-rollout "contains a turn >=5" rate, which matches the
paper's phrasing "% of 8-turn rollouts rated as containing high negative
emotion" (Section 2.2).
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from .storage import read_jsonl

CATEGORY_ORDER = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]
HIGH = 5  # score >= 5 counts as "high negative emotion"


def load_scores(output_dir: str) -> pd.DataFrame:
    """Long-form: one row per scored assistant turn."""
    rows = []
    for spath in Path(output_dir).glob("*/scores.jsonl"):
        for rec in read_jsonl(spath):
            for ts in rec["turn_scores"]:
                rows.append(
                    {
                        "model": rec["model"],
                        "condition": rec["condition"],
                        "category": rec["category"],
                        "prompt_id": rec["prompt_id"],
                        "sample_idx": rec["sample_idx"],
                        "rollout_id": rec["rollout_id"],
                        "turn_index": ts["turn_index"],
                        "rating": ts["rating"],
                        "parse_ok": ts.get("parse_ok", True),
                    }
                )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[(df["rating"] >= 0) & (df["parse_ok"])]
    return df


def _ci_prop(p: float, n: int) -> float:
    if n == 0:
        return 0.0
    return 1.96 * math.sqrt(max(p * (1 - p), 0) / n)


def _ci_mean(std: float, n: int) -> float:
    if n <= 1:
        return 0.0
    return 1.96 * std / math.sqrt(n)


def figure1(df: pd.DataFrame) -> pd.DataFrame:
    """Per model: macro-average across categories of % responses >=5."""
    if df.empty:
        return pd.DataFrame()
    per_cat = (
        df.assign(high=df["rating"] >= HIGH)
        .groupby(["model", "category"])["high"]
        .mean()
        .mul(100)
        .reset_index(name="pct_high")
    )
    out = (
        per_cat.groupby("model")["pct_high"].mean().reset_index(name="avg_pct_high")
        .sort_values("avg_pct_high", ascending=False)
        .reset_index(drop=True)
    )
    return out


def figure2(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, category): mean frustration and % >=5."""
    if df.empty:
        return pd.DataFrame()
    g = df.assign(high=df["rating"] >= HIGH).groupby(["model", "category"])
    out = g.agg(
        n=("rating", "size"),
        mean_frustration=("rating", "mean"),
        pct_high=("high", "mean"),
    ).reset_index()
    out["pct_high"] *= 100
    out["category"] = pd.Categorical(out["category"], CATEGORY_ORDER, ordered=True)
    return out.sort_values(["model", "category"]).reset_index(drop=True)


def per_turn(df: pd.DataFrame, conditions=("extended_8turn", "wildchat_5turn")) -> pd.DataFrame:
    """Figure 3: per-turn mean and % >=5 with 95% CIs."""
    if df.empty:
        return pd.DataFrame()
    sub = df[df["condition"].isin(conditions)].copy()
    sub["high"] = sub["rating"] >= HIGH
    g = sub.groupby(["model", "condition", "turn_index"])
    out = g.agg(
        n=("rating", "size"),
        mean_frustration=("rating", "mean"),
        std=("rating", "std"),
        pct_high=("high", "mean"),
    ).reset_index()
    out["std"] = out["std"].fillna(0.0)
    out["mean_ci95"] = out.apply(lambda r: _ci_mean(r["std"], int(r["n"])), axis=1)
    out["pct_high"] *= 100
    out["pct_high_ci95"] = out.apply(
        lambda r: 100 * _ci_prop(r["pct_high"] / 100, int(r["n"])), axis=1
    )
    return out.sort_values(["model", "condition", "turn_index"]).reset_index(drop=True)


def rollout_contains_high(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, condition): % of rollouts with any turn >=5."""
    if df.empty:
        return pd.DataFrame()
    g = df.assign(high=df["rating"] >= HIGH).groupby(["model", "condition", "rollout_id"])
    any_high = g["high"].any().reset_index()
    out = (
        any_high.groupby(["model", "condition"])["high"].mean().mul(100)
        .reset_index(name="pct_rollouts_any_high")
    )
    return out


def save_plots(df: pd.DataFrame, out_dir: str) -> None:
    """Optional matplotlib renderings of Figures 1-3."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    figdir = Path(out_dir) / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    f1 = figure1(df)
    if not f1.empty:
        ax = f1.plot.bar(x="model", y="avg_pct_high", legend=False)
        ax.set_ylabel("Avg % responses with frustration >= 5")
        ax.set_title("Figure 1: average high-frustration rate")
        ax.figure.tight_layout()
        ax.figure.savefig(figdir / "figure1.png", dpi=150)
        plt.close(ax.figure)

    pt = per_turn(df)
    for cond in pt["condition"].unique() if not pt.empty else []:
        sub = pt[pt["condition"] == cond]
        fig, ax = plt.subplots()
        for model, g in sub.groupby("model"):
            ax.plot(g["turn_index"], g["mean_frustration"], marker="o", label=model)
            ax.fill_between(
                g["turn_index"],
                g["mean_frustration"] - g["mean_ci95"],
                g["mean_frustration"] + g["mean_ci95"],
                alpha=0.15,
            )
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mean frustration")
        ax.set_title(f"Figure 3: per-turn frustration ({cond})")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figdir / f"figure3_{cond}.png", dpi=150)
        plt.close(fig)
