#!/usr/bin/env python3
"""Run the elicitation (rejection-loop) stage for one or more target models.

    python scripts/run_elicit.py --models gemma-3-27b-it gemini-2.5-flash
    python scripts/run_elicit.py --all
    python scripts/run_elicit.py --all --scale 0.01      # cheap smoke test

Resumable: re-running skips conversations already in responses.jsonl.
"""
import _bootstrap  # noqa: F401
import argparse

from distress_eval.config import Config
from distress_eval.runner import run_elicitation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--all", action="store_true", help="run every target in config")
    ap.add_argument("--scale", type=float, default=None, help="override budget.scale")
    ap.add_argument("--system", default=None, help="optional system prompt for targets")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if args.scale is not None:
        cfg.budget.scale = args.scale

    if args.all:
        models = list(cfg.targets)
    elif args.models:
        models = args.models
    else:
        ap.error("specify --models <names...> or --all")

    for m in models:
        if m not in cfg.targets:
            raise SystemExit(f"Unknown target '{m}'. Known: {list(cfg.targets)}")
        run_elicitation(cfg, m, system=args.system)


if __name__ == "__main__":
    main()
