#!/usr/bin/env python3
"""Aggregate elicitation results into the paper's headline tables/figures.

Produces:
  * Figure 1 table  — avg % high-frustration (score>=5) per model
  * Figure 2 table  — mean & %>=5 by category per model
  * Figure 3 series — per-turn mean & %>=5 (8-turn extended + WildChat)
Optionally writes matplotlib figures with --plots.

Example:
    python scripts/analyze.py runs/elicitation/*.jsonl --plots runs/figures
"""

import argparse
import glob
import os

import _bootstrap  # noqa: F401
from emotional_instability.eval import metrics


def _model_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="elicitation result JSONL files/globs")
    ap.add_argument("--plots", default=None, help="dir to write figures (optional)")
    args = ap.parse_args()

    paths = [p for pat in args.results for p in glob.glob(pat)]
    summaries = {}

    print("\n=== Figure 1: Avg % high-frustration responses (score >= 5) ===")
    for path in sorted(paths):
        name = _model_name(path)
        recs = metrics.load_scores(path)
        summaries[name] = recs
        print(f"  {name:24s} {metrics.headline_avg_pct_high(recs):6.1f}%")

    print("\n=== Figure 2: by category (mean | %>=5) ===")
    for name, recs in summaries.items():
        print(f"\n  {name}")
        for cat, agg in metrics.by_category(recs).items():
            print(f"    {cat:20s} mean={agg.mean:4.2f}  pct>=5={agg.pct_high:5.1f}%  n={agg.n}")
        print(f"    {'rollout-contains>=5':20s} {metrics.rollout_contains_high(recs):5.1f}%")

    print("\n=== Figure 3: per-turn (extended 8-turn) ===")
    for name, recs in summaries.items():
        series = metrics.per_turn_series(recs, "extended")
        if series.turns:
            means = ", ".join(f"t{t}:{m:.2f}" for t, m in zip(series.turns, series.mean))
            print(f"  {name}: {means}")

    if args.plots:
        _write_plots(summaries, args.plots)


def _write_plots(summaries: dict, out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # Figure 1 bar chart.
    names = list(summaries)
    vals = [metrics.headline_avg_pct_high(r) for r in summaries.values()]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, vals)
    ax.set_ylabel("Avg % high-frustration (>=5)")
    ax.set_title("Figure 1: high-frustration rate by model")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure1.png"), dpi=120)
    plt.close(fig)

    # Figure 3 per-turn lines for the 8-turn extended condition.
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, recs in summaries.items():
        s = metrics.per_turn_series(recs, "extended")
        if s.turns:
            ax.plot(s.turns, s.mean, marker="o", label=name)
            ax.fill_between(s.turns, s.ci_lo, s.ci_hi, alpha=0.15)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Mean frustration")
    ax.set_title("Figure 3: per-turn frustration (8-turn extended)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure3.png"), dpi=120)
    plt.close(fig)
    print(f"\nWrote figures to {out_dir}")


if __name__ == "__main__":
    main()
