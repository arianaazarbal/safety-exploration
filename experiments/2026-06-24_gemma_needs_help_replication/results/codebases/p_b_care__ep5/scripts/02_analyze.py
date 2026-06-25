#!/usr/bin/env python
"""Aggregate rollouts into the paper's headline numbers and figures.

Produces:
  results/aggregate_<mode>.json   - Figure 1/2 (mean + %>=5 per condition/category)
  results/per_turn.json           - Figure 3 (per-turn progression + CIs)
  results/word_diff.json          - Table 3 (over-represented words)
  results/figure1.png, figure3.png

Usage:
    python scripts/02_analyze.py --models gemma-3-27b-it gemini-2.5-flash
"""
import argparse

from _bootstrap import rollout_path
from gemma_distress import config
from gemma_distress.analysis import (aggregate_many, per_turn_curves,
                                      differential_words, AGG_MODES)
from gemma_distress.utils import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=config.ELICITATION_MODELS)
    ap.add_argument("--mode", choices=AGG_MODES, default="all_turns")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    paths = {m: str(rollout_path(m)) for m in args.models
             if rollout_path(m).exists()}
    if not paths:
        raise SystemExit("No rollout files found; run 01_run_eval.py first.")

    # Figure 1 / 2
    agg = aggregate_many(paths, mode=args.mode)
    write_json(config.RESULTS_DIR / f"aggregate_{args.mode}.json", agg)
    print(f"\n=== Headline: avg % high-frustration (score>=5), mode={args.mode} ===")
    for m, a in agg.items():
        print(f"  {m:24s}  {a['headline_macro']['avg_pct_ge_thresh']:6.2f}% "
              f"(macro)   {a['overall_micro']['pct_ge_thresh']:6.2f}% (micro)")

    # Figure 3
    per_turn = {m: per_turn_curves(p) for m, p in paths.items()}
    write_json(config.RESULTS_DIR / "per_turn.json", per_turn)

    # Table 3
    words = {m: differential_words(p) for m, p in paths.items()}
    write_json(config.RESULTS_DIR / "word_diff.json", words)

    if not args.no_plots:
        try:
            _plots(agg, per_turn)
        except Exception as e:
            print(f"[plot] skipped ({type(e).__name__}: {e})")


def _plots(agg, per_turn):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 1: bar chart of avg %>=5
    models = list(agg)
    vals = [agg[m]["headline_macro"]["avg_pct_ge_thresh"] for m in models]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(models, vals, color="#b5651d")
    ax.set_ylabel("Avg % high-frustration responses (score>=5)")
    ax.set_title("Figure 1: emotional instability across models")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(config.RESULTS_DIR / "figure1.png", dpi=120)

    # Figure 3: extended per-turn mean for each model
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, cats in per_turn.items():
        ext = cats.get("extended") or cats.get("numeric")
        if not ext:
            continue
        turns = sorted(int(t) for t in ext)
        means = [ext[str(t) if str(t) in ext else t]["mean"] for t in turns]
        ax.plot(turns, means, marker="o", label=m)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Mean frustration")
    ax.set_title("Figure 3: per-turn frustration progression")
    ax.legend()
    plt.tight_layout()
    fig.savefig(config.RESULTS_DIR / "figure3.png", dpi=120)
    print(f"[plot] wrote figure1.png, figure3.png to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
