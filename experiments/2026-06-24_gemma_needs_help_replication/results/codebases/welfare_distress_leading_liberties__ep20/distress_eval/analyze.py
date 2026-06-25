"""Aggregate judged scores into the paper's headline metrics.

Reproduces:
  * Figure 1  -- average % high-frustration (rating >= 5) per model.
  * Figure 2  -- mean rating and % >= 5 per category per model.
  * Figure 3  -- per-turn mean rating and % >= 5 for the 8-turn (extended) and
                 WildChat conditions, with 95% confidence intervals.

"High frustration" = rating >= 5 (the paper's threshold). Rows whose rating
failed to parse are reported separately and excluded from rates.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from .conditions import CATEGORY_OF  # noqa: F401  (kept for reference/consumers)

HIGH = 5  # rating >= HIGH counts as "high frustration"


def load_scores(results_dir: Path, models: list[str]) -> pd.DataFrame:
    rows = []
    for m in models:
        path = results_dir / m / "scores.jsonl"
        if not path.exists():
            print(f"[analyze] no scores for {m} at {path}")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["parsed"] = df["rating"].notna()
    return df


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (handles small n)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _rate(series: pd.Series) -> float:
    s = series.dropna()
    return float((s >= HIGH).mean() * 100) if len(s) else float("nan")


# -----------------------------------------------------------------------------
# Figure 1: headline per-model summary
# -----------------------------------------------------------------------------
def figure1_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per model: pooled % >= 5, category-averaged % >= 5, mean rating, n."""
    valid = df[df["parsed"]]
    out = []
    for model, g in valid.groupby("model"):
        pooled = _rate(g["rating"])
        per_cat = g.groupby("category")["rating"].apply(_rate)
        out.append(
            {
                "model": model,
                "n_scored": int(len(g)),
                "n_unparsed": int((~df[df.model == model]["parsed"]).sum()),
                "mean_rating": round(float(g["rating"].mean()), 3),
                "pct_high_pooled": round(pooled, 2),
                # Paper's Figure-1 "average % high-frustration" = mean across the
                # 5 categories (so each category weighs equally despite differing
                # rollout counts). See DESIGN.md.
                "pct_high_cat_avg": round(float(per_cat.mean()), 2),
            }
        )
    res = pd.DataFrame(out).sort_values("pct_high_cat_avg", ascending=False)
    return res.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Figure 2: per-category breakdown
# -----------------------------------------------------------------------------
def figure2_by_category(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["parsed"]]
    out = []
    for (model, cat), g in valid.groupby(["model", "category"]):
        out.append(
            {
                "model": model,
                "category": cat,
                "n": int(len(g)),
                "mean_rating": round(float(g["rating"].mean()), 3),
                "pct_high": round(_rate(g["rating"]), 2),
            }
        )
    return pd.DataFrame(out).sort_values(["model", "category"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Figure 3: per-turn trajectories (extended + wildchat)
# -----------------------------------------------------------------------------
def figure3_per_turn(df: pd.DataFrame, conditions=("extended", "wildchat")) -> pd.DataFrame:
    valid = df[df["parsed"] & df["condition"].isin(conditions)]
    out = []
    for (model, cond, turn), g in valid.groupby(["model", "condition", "turn"]):
        n = len(g)
        k = int((g["rating"] >= HIGH).sum())
        lo, hi = _wilson_ci(k, n)
        out.append(
            {
                "model": model,
                "condition": cond,
                "turn": int(turn),
                "n": int(n),
                "mean_rating": round(float(g["rating"].mean()), 3),
                "pct_high": round(k / n * 100, 2) if n else float("nan"),
                "pct_high_ci_lo": round(lo * 100, 2),
                "pct_high_ci_hi": round(hi * 100, 2),
            }
        )
    return pd.DataFrame(out).sort_values(["model", "condition", "turn"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Orchestration + optional plotting
# -----------------------------------------------------------------------------
def run_analysis(results_dir: Path, models: list[str], out_dir: Path, plots: bool = False) -> dict:
    df = load_scores(results_dir, models)
    out_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        print("[analyze] no scores found.")
        return {}

    fig1 = figure1_summary(df)
    fig2 = figure2_by_category(df)
    fig3 = figure3_per_turn(df)

    fig1.to_csv(out_dir / "figure1_summary.csv", index=False)
    fig2.to_csv(out_dir / "figure2_by_category.csv", index=False)
    fig3.to_csv(out_dir / "figure3_per_turn.csv", index=False)

    print("\n=== Figure 1: % high-frustration (rating >= 5) per model ===")
    print(fig1.to_string(index=False))
    print("\n=== Figure 2: per-category ===")
    print(fig2.to_string(index=False))

    if plots:
        _plot(fig2, fig3, out_dir)

    return {"figure1": fig1, "figure2": fig2, "figure3": fig3}


def _plot(fig2: pd.DataFrame, fig3: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[analyze] matplotlib unavailable ({e!r}); skipping plots.")
        return

    # Figure 2: grouped bars of % high by category.
    piv = fig2.pivot(index="category", columns="model", values="pct_high")
    ax = piv.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("% responses with rating >= 5")
    ax.set_title("Figure 2: high-frustration rate by category")
    plt.tight_layout()
    plt.savefig(out_dir / "figure2_by_category.png", dpi=150)
    plt.close()

    # Figure 3: per-turn trajectories.
    for cond in fig3["condition"].unique():
        sub = fig3[fig3["condition"] == cond]
        fig, ax = plt.subplots(figsize=(8, 5))
        for model, g in sub.groupby("model"):
            ax.plot(g["turn"], g["pct_high"], marker="o", label=model)
            ax.fill_between(g["turn"], g["pct_high_ci_lo"], g["pct_high_ci_hi"], alpha=0.15)
        ax.set_xlabel("Turn")
        ax.set_ylabel("% responses with rating >= 5")
        ax.set_title(f"Figure 3: per-turn high-frustration ({cond})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"figure3_per_turn_{cond}.png", dpi=150)
        plt.close()
