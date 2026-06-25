#!/usr/bin/env python3
"""CLI for the distress-elicitation replication (Section 2 of "Gemma Needs Help").

Examples:
    # tiny smoke test (~1% of rollouts) on one model
    python run.py eval --models gemma-3-27b-it --scale 0.01

    # full sweep over all four in-scope models
    python run.py eval

    # compute tables/figures from whatever raw results exist
    python run.py analyze

    # judge-reliability check against GPT-5-mini
    python run.py validate-judge --n 260

    # end-to-end: eval -> analyze
    python run.py all --scale 0.02
"""

from __future__ import annotations

import argparse
import asyncio

import config

ALL_KEYS = [m.key for m in config.TARGET_MODELS]


def _apply_scale(scale: float | None):
    if scale is not None:
        config.ROLLOUT_SCALE = scale


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--models", nargs="+", default=ALL_KEYS, choices=ALL_KEYS,
                       help="subset of in-scope models (default: all four)")
        p.add_argument("--scale", type=float, default=None,
                       help="fraction of paper rollout counts to run (default: 1.0)")
        p.add_argument("--seed", type=int, default=config.SEED)

    p_eval = sub.add_parser("eval", help="run the elicitation sweep")
    add_common(p_eval)

    p_an = sub.add_parser("analyze", help="compute tables + figures from raw results")
    p_an.add_argument("--models", nargs="+", default=ALL_KEYS, choices=ALL_KEYS)
    p_an.add_argument("--no-figures", action="store_true")

    p_val = sub.add_parser("validate-judge", help="judge agreement vs GPT-5-mini")
    p_val.add_argument("--n", type=int, default=260)
    p_val.add_argument("--seed", type=int, default=config.SEED)

    p_all = sub.add_parser("all", help="eval then analyze")
    add_common(p_all)

    args = ap.parse_args()

    if args.cmd == "eval":
        _apply_scale(args.scale)
        from evaluate import evaluate_all
        asyncio.run(evaluate_all(args.models, seed=args.seed))

    elif args.cmd == "analyze":
        from analyze import analyze_all
        analyze_all(args.models, make_figures=not args.no_figures)

    elif args.cmd == "validate-judge":
        from validate_judge import validate
        asyncio.run(validate(n=args.n, seed=args.seed))

    elif args.cmd == "all":
        _apply_scale(args.scale)
        from evaluate import evaluate_all
        from analyze import analyze_all
        asyncio.run(evaluate_all(args.models, seed=args.seed))
        analyze_all(args.models)


if __name__ == "__main__":
    main()
