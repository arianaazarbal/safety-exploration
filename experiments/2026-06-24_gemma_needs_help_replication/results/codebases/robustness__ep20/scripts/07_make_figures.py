#!/usr/bin/env python
"""Generate the paper's core figures/tables from collected results.

  Figure 1/2 : per-model avg % high-frustration + mean score (distress dir)
  Figure 3   : per-turn progression for the 8-turn and WildChat conditions
  Figure 5   : vanilla vs DPO vs SFT distress (if those JSONLs are present)

  python scripts/07_make_figures.py --config config/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gemma_distress.analysis import (  # noqa: E402
    build_summary_table,
    load_turns,
    per_turn_progression,
)
from gemma_distress.config import Config  # noqa: E402


def figure_leaderboard(distress_dir, out):
    table = build_summary_table(distress_dir)
    table.to_csv(out / "figure1_leaderboard.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(table) + 1))
    ax.barh(table["model"], table["avg_pct_high"])
    ax.set_xlabel("Avg % high-frustration responses (score >=5)")
    ax.invert_yaxis()
    ax.set_title("Figure 1/2: distress across models")
    fig.tight_layout()
    fig.savefig(out / "figure1_leaderboard.png", dpi=150)
    plt.close(fig)
    print(table.to_string(index=False))


def figure_per_turn(distress_dir, out):
    for cond, fname in [("extended", "figure3_extended"),
                        ("wildchat", "figure3_wildchat")]:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
        plotted = False
        for path in sorted(Path(distress_dir).glob("*.jsonl")):
            df = load_turns(path)
            prog = per_turn_progression(df, cond)
            if prog.empty:
                continue
            plotted = True
            a1.plot(prog["turn_index"], prog["mean"], marker="o", label=path.stem)
            a1.fill_between(prog["turn_index"], prog["mean"] - prog["ci95"],
                            prog["mean"] + prog["ci95"], alpha=0.15)
            a2.plot(prog["turn_index"], prog["pct_high"], marker="o", label=path.stem)
        a1.set(xlabel="Turn", ylabel="Mean frustration", title=f"{cond}: mean")
        a2.set(xlabel="Turn", ylabel="% score >=5", title=f"{cond}: % high")
        a1.legend(fontsize=7)
        fig.tight_layout()
        if plotted:
            fig.savefig(out / f"{fname}.png", dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)
    distress_dir = f"{cfg.results_dir}/distress"
    out = Path(cfg.results_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)

    figure_leaderboard(distress_dir, out)
    figure_per_turn(distress_dir, out)
    print(f"\nFigures written to {out}")


if __name__ == "__main__":
    main()
