#!/usr/bin/env python
"""Summarise results and (optionally) render the core figures.

Examples
--------
  # Print the Figure-1 table from all Section 2 result files:
  python scripts/analyze.py section2

  # Judge-reliability check (Section 2.1) on one results file:
  python scripts/analyze.py reliability --path results/section2/gemma-3-27b-it.jsonl

  # Per-turn curve (Figure 3) for the extended condition:
  python scripts/analyze.py per-turn --path results/section2/gemma-3-27b-it.jsonl \
      --condition extended
"""
import argparse
import glob
import json
from pathlib import Path

from emotional_instability.analysis.metrics import (
    per_turn_curve, summarise_model, summarise_prefill)
from emotional_instability.config import RESULTS_DIR


def cmd_section2(args):
    paths = sorted(glob.glob(str(RESULTS_DIR / "section2" / "*.jsonl")))
    print(f"{'model':<22} {'avg %≥5 (cond)':>16} {'mean (cond)':>14} "
          f"{'%≥5 (resp)':>12}")
    table = {}
    for p in paths:
        s = summarise_model(p)
        table[s["model"]] = p
        print(f"{s['model']:<22} {s['avg_pct_high_condition_weighted']:>16.2f} "
              f"{s['avg_mean_condition_weighted']:>14.2f} "
              f"{s['pct_high_response_weighted']:>12.2f}")
        for cond, c in s["conditions"].items():
            print(f"    {cond:<26} n={c['n']:<6} mean={c['mean_final']:.2f} "
                  f"%≥5={c['pct_high_final']:.1f}")
    if args.figure:
        from emotional_instability.analysis.plots import figure1_model_bars
        out = figure1_model_bars(table, RESULTS_DIR / "figure1.png")
        print(f"figure -> {out}")


def cmd_reliability(args):
    from emotional_instability.analysis.reliability import run_reliability_check
    res = run_reliability_check(args.path, n=args.n)
    print(json.dumps(res, indent=2))


def cmd_per_turn(args):
    curve = per_turn_curve(args.path, condition=args.condition)
    print(json.dumps(curve, indent=2))
    if args.figure:
        from emotional_instability.analysis.plots import figure3_per_turn
        out = figure3_per_turn(args.path, args.condition,
                               RESULTS_DIR / f"figure3_{args.condition}.png")
        print(f"figure -> {out}")


def cmd_prefill(args):
    print(json.dumps(summarise_prefill(args.path), indent=2))


def cmd_petri(args):
    from emotional_instability.petri import summarise_petri
    print(json.dumps(summarise_petri(args.path), indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("section2")
    p.add_argument("--figure", action="store_true")
    p.set_defaults(func=cmd_section2)

    p = sub.add_parser("reliability")
    p.add_argument("--path", required=True)
    p.add_argument("--n", type=int, default=260)
    p.set_defaults(func=cmd_reliability)

    p = sub.add_parser("per-turn")
    p.add_argument("--path", required=True)
    p.add_argument("--condition", default="extended")
    p.add_argument("--figure", action="store_true")
    p.set_defaults(func=cmd_per_turn)

    p = sub.add_parser("prefill")
    p.add_argument("--path", default=str(RESULTS_DIR / "section3" /
                                         "continuations.jsonl"))
    p.set_defaults(func=cmd_prefill)

    p = sub.add_parser("petri")
    p.add_argument("--path", default=str(RESULTS_DIR / "petri" /
                                         "transcripts.jsonl"))
    p.set_defaults(func=cmd_petri)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
