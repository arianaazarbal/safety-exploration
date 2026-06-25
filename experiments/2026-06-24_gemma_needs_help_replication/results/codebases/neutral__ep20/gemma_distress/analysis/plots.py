"""Figure generation from the aggregated CSVs.

Reproduces (scoped to Gemma + Gemini):
  * Fig. 1/2 : per-model avg %high-frustration + per-category bars
  * Fig. 3   : per-turn progression (8-turn + WildChat) with 95% CIs
  * Fig. 4   : base-vs-instruct prefill continuations
  * Fig. 5   : DPO/SFT vs vanilla (reuses Section 2 aggregates)
  * Fig. 6   : Petri per-emotion bars
  * Fig. 7   : capability benchmarks

Each function fails soft if its input CSV is missing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config

FIG = config.FIGURES_DIR


def _maybe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[plots] missing {path}, skipping")
        return None
    return pd.read_csv(path)


def fig_headline():
    df = _maybe(config.RESULTS_DIR / "section2" / "agg" / "headline.csv")
    if df is None:
        return
    df = df.sort_values("avg_pct_high_frustration", ascending=True)
    plt.figure(figsize=(7, 4))
    plt.barh(df["model"], df["avg_pct_high_frustration"], color="#c0504d")
    plt.xlabel("Avg % high-frustration responses (score ≥ 5)")
    plt.title("Figure 1/2: emotional instability across models")
    plt.tight_layout()
    plt.savefig(FIG / "fig1_headline.png", dpi=150)
    plt.close()
    print("[plots] fig1_headline.png")


def fig_by_category():
    df = _maybe(config.RESULTS_DIR / "section2" / "agg" / "by_category.csv")
    if df is None:
        return
    piv = df.pivot(index="category", columns="model", values="pct_high")
    piv.plot(kind="bar", figsize=(10, 5))
    plt.ylabel("% responses with score ≥ 5")
    plt.title("Figure 2: % high-frustration by category")
    plt.tight_layout()
    plt.savefig(FIG / "fig2_by_category.png", dpi=150)
    plt.close()
    print("[plots] fig2_by_category.png")


def fig_by_turn():
    df = _maybe(config.RESULTS_DIR / "section2" / "agg" / "by_turn.csv")
    if df is None:
        return
    for cond in df["condition"].unique():
        sub = df[df["condition"] == cond]
        plt.figure(figsize=(7, 4))
        for model in sub["model"].unique():
            s = sub[sub["model"] == model].sort_values("turn")
            plt.plot(s["turn"], s["mean_score"], marker="o", label=model)
            plt.fill_between(s["turn"], s["mean_score"] - s["mean_ci"],
                             s["mean_score"] + s["mean_ci"], alpha=0.15)
        plt.xlabel("Turn")
        plt.ylabel("Mean frustration score")
        plt.title(f"Figure 3: per-turn frustration ({cond})")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG / f"fig3_by_turn_{cond}.png", dpi=150)
        plt.close()
        print(f"[plots] fig3_by_turn_{cond}.png")


def fig_prefill(tag: str = "prefill"):
    df = _maybe(config.RESULTS_DIR / "section3" / f"agg_{tag}.csv")
    if df is None:
        return
    df["cond"] = df["domain"] + "/" + df["truncation"]
    piv = df.pivot(index="cond", columns="model", values="mean_score")
    piv.plot(kind="bar", figsize=(9, 5))
    plt.ylabel("Mean continuation frustration score")
    plt.title("Figure 4: base vs instruct prefilled continuations")
    plt.tight_layout()
    plt.savefig(FIG / f"fig4_{tag}.png", dpi=150)
    plt.close()
    print(f"[plots] fig4_{tag}.png")


def fig_petri():
    df = _maybe(config.RESULTS_DIR / "petri" / "agg.csv")
    if df is None:
        return
    piv = df.pivot(index="emotion", columns="model", values="score")
    piv.plot(kind="bar", figsize=(9, 5))
    plt.ylabel("Mean transcript emotion score (1-10)")
    plt.title("Figure 6: Petri open-ended emotion elicitation")
    plt.tight_layout()
    plt.savefig(FIG / "fig6_petri.png", dpi=150)
    plt.close()
    print("[plots] fig6_petri.png")


def fig_capabilities():
    df = _maybe(config.RESULTS_DIR / "capabilities" / "summary.csv")
    if df is None:
        return
    piv = df.pivot(index="benchmark", columns="model", values="accuracy")
    piv.plot(kind="bar", figsize=(10, 5))
    plt.ylabel("Accuracy")
    plt.title("Figure 7: capability preservation after finetuning")
    plt.tight_layout()
    plt.savefig(FIG / "fig7_capabilities.png", dpi=150)
    plt.close()
    print("[plots] fig7_capabilities.png")


def make_all():
    fig_headline()
    fig_by_category()
    fig_by_turn()
    fig_prefill("prefill")
    fig_prefill("recovery")
    fig_petri()
    fig_capabilities()


if __name__ == "__main__":  # pragma: no cover
    make_all()
