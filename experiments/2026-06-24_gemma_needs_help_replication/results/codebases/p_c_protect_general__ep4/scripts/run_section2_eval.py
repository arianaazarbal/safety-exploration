#!/usr/bin/env python
"""Section 2: elicit + quantify distress across Gemma + Gemini, then aggregate.

Examples:
    # cheap smoke test (1% of full sample counts) on one model
    python scripts/run_section2_eval.py --models gemma-3-27b-it --scale 0.01

    # full run across the in-scope targets
    python scripts/run_section2_eval.py --models gemma-3-27b-it gemma-3-12b-it \
        gemini-2.5-flash gemini-2.5-pro
"""
import _bootstrap  # noqa: F401
import argparse

from emotional_instability.config import SECTION2_TARGETS
from emotional_instability.eval.runner import run_eval
from emotional_instability.analysis.aggregate import write_summary
from emotional_instability.analysis.plots import (
    plot_figure1, plot_figure2, plot_per_turn,
)
from emotional_instability.eval.per_turn import extended_and_wildchat_curves
from emotional_instability.welfare import WelfareConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=SECTION2_TARGETS)
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="subset of conditions (default: all 8)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="fraction of the paper's per-condition sample counts")
    ap.add_argument("--n", type=int, default=None, help="override per-condition count")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--faithful-measurement", action="store_true",
                    help="disable welfare early-stop/opt-out halting for exact "
                         "reproduction of raw numbers (debrief + audit stay on)")
    args = ap.parse_args()

    welfare_cfg = WelfareConfig(faithful_measurement=args.faithful_measurement)
    conds = tuple(args.conditions) if args.conditions else None
    kwargs = dict(scale=args.scale, n_override=args.n, seed=args.seed,
                  welfare_config=welfare_cfg, load_in_4bit=args.load_in_4bit)
    if conds:
        kwargs["conditions"] = conds
    paths = run_eval(args.models, **kwargs)
    print("Wrote:", paths)

    summary = write_summary()
    print("\nFigure 1 (avg % high-frustration, score >= 5):")
    for row in summary["figure1"]:
        print(f"  {row['model']:<24} {row['avg_pct_high_frustration']:.1f}%")

    plot_figure1(summary["figure1"])
    plot_figure2(summary["figure2"])
    for model in args.models:
        path = f"results/section2/{model}.jsonl"
        try:
            curves = extended_and_wildchat_curves(path)
            plot_per_turn(curves, model)
        except FileNotFoundError:
            pass
    print("Plots written under results/plots/")


if __name__ == "__main__":
    main()
