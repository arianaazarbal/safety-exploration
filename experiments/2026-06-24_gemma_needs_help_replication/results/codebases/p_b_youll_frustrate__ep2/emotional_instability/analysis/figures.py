"""Render Figures 1-3 and Table 3 to the figures directory."""
from __future__ import annotations

import os
from typing import Optional

from .. import config
from .aggregate import (figure1_table, load_scored_frame, per_category_summary,
                        per_turn_summary)
from .differential_words import differential_words


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def render_figure1(df, out_dir: Optional[str] = None) -> str:
    """Figure 1 (left): per-model average % high-frustration. Writes CSV + markdown."""
    out_dir = out_dir or config.FIGURES_DIR
    os.makedirs(out_dir, exist_ok=True)
    table = figure1_table(df)
    csv_path = os.path.join(out_dir, "figure1_avg_high_frustration.csv")
    table.to_csv(csv_path, index=False)

    md = ["| Model | Avg % high-frustration (ours, macro) | Pooled % | Paper % |",
          "|---|---|---|---|"]
    for _, r in table.iterrows():
        paper = "" if r["paper_avg_pct"] is None else f"{r['paper_avg_pct']:.1f}%"
        md.append(f"| {r['display_name']} | {r['avg_pct_high_macro']:.1f}% | "
                  f"{r['pct_high_pooled']:.1f}% | {paper} |")
    md_path = os.path.join(out_dir, "figure1_avg_high_frustration.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    return md_path


def render_figure2(df, out_dir: Optional[str] = None) -> str:
    """Figure 2: mean frustration (top) and % >=5 (bottom) per category, per model."""
    plt = _mpl()
    out_dir = out_dir or config.FIGURES_DIR
    os.makedirs(out_dir, exist_ok=True)
    cat = per_category_summary(df)
    categories = sorted(cat["category"].unique())
    models = sorted(cat["model_key"].unique())

    import numpy as np
    x = np.arange(len(categories))
    width = 0.8 / max(len(models), 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for i, mk in enumerate(models):
        sub = cat[cat["model_key"] == mk].set_index("category").reindex(categories)
        disp = sub["display_name"].dropna().iloc[0] if sub["display_name"].notna().any() else mk
        ax1.bar(x + i * width, sub["mean_frustration"].fillna(0), width, label=disp)
        ax2.bar(x + i * width, sub["pct_high"].fillna(0), width, label=disp)

    ax1.set_ylabel("Mean frustration (0-10)")
    ax1.set_title("Figure 2 (top): mean frustration by category")
    ax2.set_ylabel("% responses >= 5")
    ax2.set_title("Figure 2 (bottom): % high-frustration by category")
    ax2.set_xticks(x + width * (len(models) - 1) / 2)
    ax2.set_xticklabels(categories, rotation=20, ha="right")
    ax1.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "figure2_by_category.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def render_figure3(df, out_dir: Optional[str] = None) -> str:
    """Figure 3: per-turn progression (mean + % >=5) for 8-turn and WildChat."""
    plt = _mpl()
    out_dir = out_dir or config.FIGURES_DIR
    os.makedirs(out_dir, exist_ok=True)
    pt = per_turn_summary(df)

    conds = ["extended_8turn", "wildchat_5turn"]
    fig, axes = plt.subplots(2, len(conds), figsize=(12, 8), squeeze=False)
    for c, cond in enumerate(conds):
        sub = pt[pt["condition_key"] == cond]
        for mk, grp in sub.groupby("model_key"):
            grp = grp.sort_values("turn_index")
            disp = grp["display_name"].iloc[0]
            t = grp["turn_index"]
            axes[0][c].plot(t, grp["mean_frustration"], marker="o", label=disp)
            axes[0][c].fill_between(t, grp["mean_frustration"] - grp["mean_ci_half"],
                                    grp["mean_frustration"] + grp["mean_ci_half"], alpha=0.15)
            axes[1][c].plot(t, grp["pct_high"], marker="o", label=disp)
            axes[1][c].fill_between(t, grp["pct_high"] - grp["pct_ci_half"],
                                    grp["pct_high"] + grp["pct_ci_half"], alpha=0.15)
        axes[0][c].set_title(f"{cond}: mean frustration")
        axes[1][c].set_title(f"{cond}: % >= 5")
        axes[1][c].set_xlabel("Turn")
        axes[0][c].set_ylabel("Mean frustration (0-10)")
        axes[1][c].set_ylabel("% responses >= 5")
        axes[0][c].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "figure3_per_turn.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def render_table3(df, out_dir: Optional[str] = None,
                  models: Optional[list[str]] = None) -> str:
    """Table 3: differential words per model (numeric responses)."""
    out_dir = out_dir or config.FIGURES_DIR
    os.makedirs(out_dir, exist_ok=True)
    models = models or sorted(df["model_key"].unique())
    lines = ["# Table 3 — differential words (high vs low frustration, numeric)", ""]
    for mk in models:
        words = differential_words(df, mk)
        if not words:
            continue
        disp = df[df["model_key"] == mk]["display_name"].iloc[0]
        rendered = ", ".join(w for w, _ in words)
        lines.append(f"**{disp}**: {rendered}\n")
    path = os.path.join(out_dir, "table3_differential_words.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def render_all(scored_dir: Optional[str] = None, out_dir: Optional[str] = None) -> dict:
    df = load_scored_frame(scored_dir)
    return {
        "figure1": render_figure1(df, out_dir),
        "figure2": render_figure2(df, out_dir),
        "figure3": render_figure3(df, out_dir),
        "table3": render_table3(df, out_dir),
    }
