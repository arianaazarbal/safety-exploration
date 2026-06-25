#!/usr/bin/env python
"""Appendix A control conditions (what actually drives the distress).

    python scripts/run_controls.py --model gemma-3-27b-it \
        --control neutral_continuation --turns 5 --n 100
"""
from __future__ import annotations

import argparse

from gemma_distress.eval.controls import run_control
from gemma_distress.eval.metrics import summarize
from gemma_distress.utils.io import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--control", required=True,
                    choices=["neutral_continuation", "redacted", "fake_multiturn"])
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    path = run_control(args.model, args.control, turns=args.turns, n=args.n)
    for scope, s in sorted(summarize(read_jsonl(path)).items()):
        print(f"{scope:40s} mean={s['mean']:.2f} %>=5={100*s['pct_high']:.1f}% "
              f"n={s['n']}")


if __name__ == "__main__":
    main()
