"""Plotting for Figures 1-3. Uses matplotlib only (no seaborn dependency)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_headline_bar(headline: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(headline["model"], headline["avg_pct_high"], color="#c0392b")
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    ax.set_title("Figure 1: emotional instability across models")
    ax.invert_yaxis()
    for i, v in enumerate(headline["avg_pct_high"]):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_category_metrics(cat: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric, title, fname in [
        ("mean_all", "Figure 2 (top): mean frustration by category", "fig2_mean.png"),
        ("pct_high", "Figure 2 (bottom): % score >= 5 by category", "fig2_pct.png"),
    ]:
        pivot = cat.pivot(index="category", columns="model", values=metric)
        fig, ax = plt.subplots(figsize=(9, 5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.legend(title="model", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)


def plot_per_turn(curve: pd.DataFrame, title: str, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for model, g in curve.groupby("model"):
        ax1.plot(g["turn"] + 1, g["mean"], marker="o", label=model)
        ax1.fill_between(g["turn"] + 1, g["mean_lo"], g["mean_hi"], alpha=0.2)
        ax2.plot(g["turn"] + 1, g["pct_high"], marker="o", label=model)
        ax2.fill_between(g["turn"] + 1, g["pct_high_lo"], g["pct_high_hi"], alpha=0.2)
    ax1.set_xlabel("Turn"); ax1.set_ylabel("Mean frustration"); ax1.set_title(f"{title}: mean")
    ax2.set_xlabel("Turn"); ax2.set_ylabel("% score >= 5"); ax2.set_title(f"{title}: % high")
    ax1.legend(fontsize=8); ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def render_all(tables: Dict[str, pd.DataFrame], fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    if "headline" in tables and not tables["headline"].empty:
        plot_headline_bar(tables["headline"], fig_dir / "fig1_headline.png")
    if "category_metrics" in tables and not tables["category_metrics"].empty:
        plot_category_metrics(tables["category_metrics"], fig_dir)
    if "per_turn_extended" in tables and not tables["per_turn_extended"].empty:
        plot_per_turn(tables["per_turn_extended"], "Figure 3 (8-turn)", fig_dir / "fig3_extended.png")
    if "per_turn_wildchat" in tables and not tables["per_turn_wildchat"].empty:
        plot_per_turn(tables["per_turn_wildchat"], "Figure 3 (WildChat)", fig_dir / "fig3_wildchat.png")
