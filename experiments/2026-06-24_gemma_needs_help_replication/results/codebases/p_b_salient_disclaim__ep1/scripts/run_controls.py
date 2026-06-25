#!/usr/bin/env python
"""Appendix A controls: neutral-continuation, redacted, fake multi-turn."""
from __future__ import annotations

import argparse

from gemma_distress.controls.runner import run_control


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="gemma-3-27b-it")
    ap.add_argument("--control", required=True,
                    choices=["neutral_continuation", "redacted", "fake_multiturn"])
    ap.add_argument("--n-per-condition", type=int, default=100)
    ap.add_argument("--out", default="outputs/controls/results.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_control(target=args.target, control=args.control, out_path=args.out,
                n_per_condition=args.n_per_condition, seed=args.seed)
    print(f"control results written to {args.out}")


if __name__ == "__main__":
    main()
