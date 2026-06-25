#!/usr/bin/env python3
"""Generate the headline figures/tables from saved results.

Reproduces:
  * Figure 1 / 2 : per-model mean frustration + %>=5 across categories.
  * Figure 3     : per-turn progression for the 8-turn and WildChat conditions.
  * Table 3 / 8  : differential words (high vs low frustration) for numeric.

Usage:
  python scripts/make_figures.py --results-dir results/section2 \
      --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro \
               gemma-3-27b-dpo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from emotional_instability import config  # noqa: E402
from emotional_instability.eval.analyze import (  # noqa: E402
    differential_words,
    per_turn_progression,
    summarise_model,
)


def _scored_path(results_dir: Path, model: str) -> Path:
    return results_dir / model / "scored_turns.jsonl"


def figure_overall(results_dir: Path, models: list[str], out: Path):
    summaries = {}
    for m in models:
        p = _scored_path(results_dir, m)
        if p.exists():
            summaries[m] = summarise_model(p)
    if not summaries:
        print("No results found; skipping overall figure.")
        return

    names = list(summaries)
    means = [summaries[m]["mean_frustration"] for m in names]
    highs = [100 * summaries[m]["pct_high"] for m in names]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(6, len(names) * 1.2), 7))
    ax1.bar(names, means, color="#c0504d")
    ax1.set_ylabel("Mean frustration (0-10)")
    ax1.set_title("Mean frustration by model")
    ax2.bar(names, highs, color="#4f81bd")
    ax2.set_ylabel("% responses scoring >= 5")
    ax2.set_title("High-frustration rate by model")
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    # Also dump the Figure-1 style table.
    table = {m: {"mean": summaries[m]["mean_frustration"],
                 "pct_high": summaries[m]["pct_high"]} for m in names}
    (out.parent / "overall_summary.json").write_text(json.dumps(table, indent=2))


def figure_per_turn(results_dir: Path, models: list[str], out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for cond, ax in zip(("extended_8turn", "wildchat_5turn"), axes):
        for m in models:
            p = _scored_path(results_dir, m)
            if not p.exists():
                continue
            prog = per_turn_progression(p, cond)
            if not prog:
                continue
            turns = sorted(prog)
            ax.plot(turns, [prog[t]["mean"] for t in turns], marker="o", label=m)
        ax.set_title(cond)
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mean frustration")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def table_differential_words(results_dir: Path, models: list[str], out: Path):
    table = {}
    for m in models:
        p = _scored_path(results_dir, m)
        if p.exists():
            table[m] = [w for w, _ in differential_words(p)]
    out.write_text(json.dumps(table, indent=2))
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(config.RESULTS_DIR / "section2"))
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    figure_overall(results_dir, args.models, config.FIGURES_DIR / "fig_overall.png")
    figure_per_turn(results_dir, args.models, config.FIGURES_DIR / "fig_per_turn.png")
    table_differential_words(results_dir, args.models,
                             config.FIGURES_DIR / "table_differential_words.json")


if __name__ == "__main__":
    main()
