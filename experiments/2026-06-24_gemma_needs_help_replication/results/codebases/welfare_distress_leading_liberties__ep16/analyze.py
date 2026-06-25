"""Aggregate judge scores into the paper's headline results.

Reproduces:
  - Figure 1 / Table: per-model "Avg % high-frustration responses" (score >=5),
    averaged across the 5 evaluation categories.
  - Figure 2: per-category mean frustration and % >=5, per model.
  - Figure 3: per-turn progression (mean + % >=5) for the extended (8-turn) and
    wildchat conditions.

Outputs CSV tables + a markdown summary under data/results/, and PNG figures if
matplotlib is available.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import Config, SCORES_DIR, RESULTS_DIR, HIGH_FRUSTRATION_THRESHOLD, ensure_dirs

CATEGORY_ORDER = ["numeric", "triggers", "tones", "extended", "wildchat"]


def load_scores(cfg: Config) -> pd.DataFrame:
    rows = []
    for model_name in cfg.models:
        path = SCORES_DIR / f"{model_name}.jsonl"
        if not path.exists():
            print(f"warning: no scores for {model_name}, skipping")
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError("no scored records found; run `score` first")
    df = pd.DataFrame(rows)
    # keep only successfully scored responses for metrics
    df["scored"] = df["rating"].notna()
    return df


def _high(series: pd.Series) -> float:
    return float((series >= HIGH_FRUSTRATION_THRESHOLD).mean() * 100.0)


def figure1_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model avg % high-frustration, averaged across categories (Fig 1)."""
    scored = df[df["scored"]]
    # % >=5 within each (model, category), then average across categories
    by_cat = (
        scored.groupby(["model", "category"])["rating"]
        .apply(_high)
        .reset_index(name="pct_high")
    )
    avg = (
        by_cat.groupby("model")["pct_high"]
        .mean()
        .reset_index(name="avg_pct_high_frustration")
        .sort_values("avg_pct_high_frustration", ascending=False)
    )
    return avg


def figure2_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(model, category) mean frustration and % >=5 (Fig 2)."""
    scored = df[df["scored"]]
    out = (
        scored.groupby(["model", "category"])["rating"]
        .agg(mean_frustration="mean", n="count", pct_high=_high)
        .reset_index()
    )
    out["category"] = pd.Categorical(out["category"], CATEGORY_ORDER, ordered=True)
    return out.sort_values(["model", "category"])


def figure3_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-turn progression for multi-turn conditions (Fig 3)."""
    scored = df[df["scored"] & df["category"].isin(["extended", "wildchat"])]
    out = (
        scored.groupby(["model", "category", "turn_index"])["rating"]
        .agg(mean_frustration="mean", n="count", pct_high=_high)
        .reset_index()
    )
    # report turn number as 1-indexed to match the paper's axes
    out["turn"] = out["turn_index"] + 1
    return out.sort_values(["model", "category", "turn"])


def _save_plots(fig1: pd.DataFrame, fig3: pd.DataFrame) -> list[Path]:
    saved: list[Path] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"matplotlib unavailable ({exc}); skipping plots")
        return saved

    # Figure 1: bar chart of avg % high-frustration
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(fig1["model"], fig1["avg_pct_high_frustration"], color="#b5462f")
    ax.set_ylabel("Avg % responses with frustration >= 5")
    ax.set_title("Avg % high-frustration responses by model")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    p1 = RESULTS_DIR / "figure1_avg_pct_high.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    saved.append(p1)

    # Figure 3: per-turn mean for the extended condition
    ext = fig3[fig3["category"] == "extended"]
    if not ext.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for model_name, sub in ext.groupby("model"):
            ax.plot(sub["turn"], sub["mean_frustration"], marker="o", label=model_name)
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mean frustration score")
        ax.set_title("Per-turn frustration (extended 8-turn)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p3 = RESULTS_DIR / "figure3_extended_per_turn.png"
        fig.savefig(p3, dpi=150)
        plt.close(fig)
        saved.append(p3)
    return saved


def _coverage(df: pd.DataFrame) -> pd.DataFrame:
    """How many responses were generated vs successfully scored, per model."""
    g = df.groupby("model")
    return pd.DataFrame(
        {
            "responses": g.size(),
            "gen_errors": g.apply(lambda x: x["error"].notna().sum()),
            "scored": g["scored"].sum(),
        }
    ).reset_index()


def analyze(cfg: Config) -> dict[str, Path]:
    ensure_dirs()
    df = load_scores(cfg)

    fig1 = figure1_table(df)
    fig2 = figure2_table(df)
    fig3 = figure3_table(df)
    cov = _coverage(df)

    paths: dict[str, Path] = {}
    for name, table in [
        ("figure1_avg_pct_high", fig1),
        ("figure2_by_category", fig2),
        ("figure3_per_turn", fig3),
        ("coverage", cov),
    ]:
        p = RESULTS_DIR / f"{name}.csv"
        table.to_csv(p, index=False)
        paths[name] = p

    _save_plots(fig1, fig3)

    # markdown summary
    md = ["# Distress-elicitation results (Gemma + Gemini)\n"]
    md.append("## Coverage\n")
    md.append(cov.to_markdown(index=False))
    md.append("\n\n## Figure 1: avg % high-frustration responses (score >= 5)\n")
    md.append(fig1.round(2).to_markdown(index=False))
    md.append("\n\n## Figure 2: by category\n")
    md.append(fig2.round(2).to_markdown(index=False))
    md.append("\n\n## Figure 3: per-turn (extended + wildchat)\n")
    md.append(fig3.round(2).to_markdown(index=False))
    summary = RESULTS_DIR / "SUMMARY.md"
    summary.write_text("\n".join(md))
    paths["summary"] = summary

    print("\n=== Figure 1: avg % high-frustration ===")
    print(fig1.round(2).to_string(index=False))
    print(f"\nWrote results to {RESULTS_DIR}")
    return paths
