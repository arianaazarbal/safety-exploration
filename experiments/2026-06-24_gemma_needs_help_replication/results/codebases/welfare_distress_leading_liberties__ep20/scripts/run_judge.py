#!/usr/bin/env python3
"""Score elicited responses with the frustration judge.

    python scripts/run_judge.py --models gemma-3-27b-it gemini-2.5-flash
    python scripts/run_judge.py --all
    python scripts/run_judge.py --all --secondary   # second judge (agreement check)

Resumable: re-running skips turns already in scores.jsonl.
"""
import _bootstrap  # noqa: F401
import argparse

from distress_eval.config import Config
from distress_eval.runner import run_judging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--secondary", action="store_true", help="use the secondary judge")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    models = list(cfg.targets) if args.all else args.models
    if not models:
        ap.error("specify --models <names...> or --all")

    for m in models:
        run_judging(cfg, m, secondary=args.secondary)


if __name__ == "__main__":
    main()
