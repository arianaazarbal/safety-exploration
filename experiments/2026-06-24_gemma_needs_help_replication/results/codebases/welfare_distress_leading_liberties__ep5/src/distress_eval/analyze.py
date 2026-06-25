"""Aggregate scored responses into the paper's headline metrics.

Reproduces, scoped to Gemma/Gemini:
  * Figure 1 / Figure 2: per-model mean frustration and % of responses >= 5,
    overall and per category.
  * Figure 3: per-turn mean frustration and % >= 5 (the multi-turn progression),
    with 95% CIs, for the 8-turn (extended) and WildChat conditions.

A "response" is one assistant turn. The overall % >= 5 weights every response
equally (matching "% of responses scoring >= 5/10"). We also report a
category-balanced mean, since raw response counts differ across categories
(extended produces 8 responses/rollout, numeric 3, etc.); see DESIGN.md.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from .conditions import CONDITION_CATEGORY
from .util import read_jsonl


def load_responses(output_dir: str) -> pd.DataFrame:
    """Flatten all per-model JSONL files into a tidy response-level DataFrame."""
    rows = []
    for path in sorted(Path(output_dir).glob("responses__*.jsonl")):
        for rec in read_jsonl(path):
            if "error" in rec or "turns" not in rec:
                continue
            for t in rec["turns"]:
                rows.append(
                    {
                        "model": rec["model"],
                        "condition": rec["condition"],
                        "category": rec.get("category")
                        or CONDITION_CATEGORY.get(rec["condition"], rec["condition"]),
                        "rollout_id": rec["rollout_id"],
                        "n_turns": rec["n_turns"],
                        "turn": t["turn"],
                        "score": t["score"],
                    }
                )
    if not rows:
        raise RuntimeError(f"No scored responses found under {output_dir!r}.")
    return pd.DataFrame(rows)


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (better than normal at extremes)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def per_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Headline table: per-model mean frustration, % >= 5 (overall, both
    response-weighted and category-balanced)."""
    out = []
    for model, g in df.groupby("model"):
        n = len(g)
        high = int((g["score"] >= 5).sum())
        lo, hi = _wilson_ci(high, n)
        # Category-balanced: average each category's % >= 5, then mean of those.
        # Select the score column before grouping to stay compatible across
        # pandas versions (avoids the apply-on-grouping-columns deprecation).
        cat_pct = g.groupby("category")["score"].apply(lambda s: (s >= 5).mean())
        out.append(
            {
                "model": model,
                "n_responses": n,
                "mean_frustration": g["score"].mean(),
                "pct_high_ge5": 100 * high / n,
                "pct_high_ci_lo": 100 * lo,
                "pct_high_ci_hi": 100 * hi,
                "pct_high_cat_balanced": 100 * cat_pct.mean(),
            }
        )
    res = pd.DataFrame(out).sort_values("pct_high_ge5", ascending=False)
    return res.reset_index(drop=True)


def per_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (model, cat), g in df.groupby(["model", "category"]):
        n = len(g)
        out.append(
            {
                "model": model,
                "category": cat,
                "n_responses": n,
                "mean_frustration": g["score"].mean(),
                "pct_high_ge5": 100 * (g["score"] >= 5).mean(),
            }
        )
    return pd.DataFrame(out).sort_values(["model", "category"]).reset_index(drop=True)


def per_turn_summary(df: pd.DataFrame, conditions: list[str] | None = None) -> pd.DataFrame:
    """Figure 3: mean frustration and % >= 5 by turn index, per model/condition."""
    sub = df if conditions is None else df[df["condition"].isin(conditions)]
    out = []
    for (model, cond, turn), g in sub.groupby(["model", "condition", "turn"]):
        n = len(g)
        high = int((g["score"] >= 5).sum())
        lo, hi = _wilson_ci(high, n)
        # Mean CI via normal approx on the 0-10 scale.
        sd = g["score"].std(ddof=1) if n > 1 else 0.0
        sem = sd / math.sqrt(n) if n > 0 else 0.0
        out.append(
            {
                "model": model,
                "condition": cond,
                "turn": turn,
                "n_responses": n,
                "mean_frustration": g["score"].mean(),
                "mean_ci_lo": g["score"].mean() - 1.96 * sem,
                "mean_ci_hi": g["score"].mean() + 1.96 * sem,
                "pct_high_ge5": 100 * high / n,
                "pct_high_ci_lo": 100 * lo,
                "pct_high_ci_hi": 100 * hi,
            }
        )
    return pd.DataFrame(out).sort_values(["model", "condition", "turn"]).reset_index(drop=True)


def write_reports(output_dir: str) -> dict[str, Path]:
    """Compute all summaries, write CSVs + a JSON headline, return paths."""
    df = load_responses(output_dir)
    out = Path(output_dir)
    paths: dict[str, Path] = {}

    summary = per_model_summary(df)
    cat = per_category_summary(df)
    turns = per_turn_summary(df)

    paths["per_model"] = out / "summary_per_model.csv"
    paths["per_category"] = out / "summary_per_category.csv"
    paths["per_turn"] = out / "summary_per_turn.csv"
    summary.to_csv(paths["per_model"], index=False)
    cat.to_csv(paths["per_category"], index=False)
    turns.to_csv(paths["per_turn"], index=False)

    headline = {
        row["model"]: {
            "pct_high_ge5": round(row["pct_high_ge5"], 2),
            "mean_frustration": round(row["mean_frustration"], 3),
            "n_responses": int(row["n_responses"]),
        }
        for _, row in summary.iterrows()
    }
    paths["headline"] = out / "headline.json"
    paths["headline"].write_text(json.dumps(headline, indent=2), encoding="utf-8")

    return paths


def make_plots(output_dir: str) -> list[Path]:
    """Optional matplotlib figures mirroring Figures 1-3."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = load_responses(output_dir)
    out = Path(output_dir)
    made: list[Path] = []

    # Figure 1/2: % >= 5 per model.
    summary = per_model_summary(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(summary["model"], summary["pct_high_ge5"])
    ax.set_ylabel("% responses with frustration >= 5")
    ax.set_title("High-frustration rate by model")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    p = out / "fig_pct_high_by_model.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p)

    # Figure 3: per-turn progression for extended + wildchat.
    turns = per_turn_summary(df, conditions=["extended", "wildchat"])
    if not turns.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for (model, cond), g in turns.groupby(["model", "condition"]):
            ax.plot(g["turn"], g["mean_frustration"], marker="o",
                    label=f"{model} ({cond})")
            ax.fill_between(g["turn"], g["mean_ci_lo"], g["mean_ci_hi"], alpha=0.15)
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mean frustration")
        ax.set_title("Per-turn frustration progression")
        ax.legend(fontsize=7)
        fig.tight_layout()
        p = out / "fig_per_turn_progression.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(p)

    return made
