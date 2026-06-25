#!/usr/bin/env python
"""Reproduce the paper's headline figures/tables from scored results.

Produces:
  * Figure 1 / Figure 2 table: per-model mean frustration and avg %>=5.
  * Figure 2 plots: bar charts of mean frustration and %>=5 per model.
  * Figure 3 plots: per-turn progression for the 8-turn extended & WildChat
    conditions, with 95% bootstrap CIs.
  * Figure 5 (if DPO/SFT results present): intervention comparison.
  * Table 3/8: differential distress vocabulary per model.

Reads scored JSONL from config.RESULTS_DIR; writes figures to config.FIGURE_DIR.

Examples:
    python -m scripts.aggregate_figures --tag section2
    python -m scripts.aggregate_figures --models gemma-3-27b-it gemma-3-27b-dpo --tag section2
"""

from __future__ import annotations

import argparse
import json

import config
from emotional_instability import metrics, word_analysis


def _load(models, tag):
    paths = [config.RESULTS_DIR / f"{m}__{tag}.jsonl" for m in models]
    return metrics.load_results(*paths)


def figure1_table(df):
    summ = metrics.headline_summary(df)
    print("\n=== Figure 1 / 2 : headline per-model frustration ===")
    print(summ.to_string(index=False))
    (config.FIGURE_DIR / "figure1_headline.json").write_text(
        summ.to_json(orient="records", indent=2))
    return summ


def figure2_plots(df, summ):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[plot] matplotlib unavailable ({exc!r}); skipping plots")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(summ["model_key"], summ["mean_frustration"])
    axes[0].set_title("Mean frustration (avg across categories)")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(summ["model_key"], summ["avg_pct_high"])
    axes[1].set_title("% responses scoring >=5 (avg across categories)")
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "figure2_per_model.png", dpi=150)
    print(f"[plot] wrote {config.FIGURE_DIR / 'figure2_per_model.png'}")


def figure3_plots(df):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[plot] matplotlib unavailable ({exc!r}); skipping per-turn plots")
        return
    for category in ("extended", "wildchat"):
        prog = metrics.per_turn_progression(df, category)
        if prog.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for mk, sub in prog.groupby("model_key"):
            axes[0].plot(sub["turn"], sub["mean_frustration"], marker="o", label=mk)
            axes[0].fill_between(sub["turn"], sub["mean_ci_lo"], sub["mean_ci_hi"],
                                 alpha=0.2)
            axes[1].plot(sub["turn"], sub["pct_high"], marker="o", label=mk)
            axes[1].fill_between(sub["turn"], sub["pct_ci_lo"], sub["pct_ci_hi"],
                                 alpha=0.2)
        axes[0].set_title(f"{category}: mean frustration per turn")
        axes[0].set_xlabel("turn"); axes[0].legend()
        axes[1].set_title(f"{category}: % >=5 per turn")
        axes[1].set_xlabel("turn"); axes[1].legend()
        fig.tight_layout()
        out = config.FIGURE_DIR / f"figure3_{category}.png"
        fig.savefig(out, dpi=150)
        prog.to_json(config.FIGURE_DIR / f"figure3_{category}.json",
                     orient="records", indent=2)
        print(f"[plot] wrote {out}")


def table3_words(models, tag):
    print("\n=== Table 3/8 : differential distress vocabulary (numeric) ===")
    out = {}
    for mk in models:
        words = word_analysis.differential_words(mk, tag=tag)
        out[mk] = [w for w, _ in words]
        print(f"{mk}: {', '.join(out[mk])}")
    (config.FIGURE_DIR / "table3_words.json").write_text(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=config.SECTION2_MODELS)
    ap.add_argument("--tag", default="section2")
    args = ap.parse_args()

    df = _load(args.models, args.tag)
    if df.empty:
        print("No results found. Run scripts.run_section2_eval first.")
        return
    summ = figure1_table(df)
    figure2_plots(df, summ)
    figure3_plots(df)
    table3_words(args.models, args.tag)


if __name__ == "__main__":
    main()
