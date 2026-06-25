#!/usr/bin/env python
"""Section 2: run the emotion-elicitation eval for the given models.

Examples:
    python scripts/01_run_eval.py --config configs/eval.smoke.yaml \
        --models gemma-3-27b-it gemini-2.5-flash
    python scripts/01_run_eval.py --config configs/eval.full.yaml \
        --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro
"""
import _bootstrap  # noqa: F401
import argparse

from distress.config import EvalConfig
from distress.eval.runner import run_eval

DEFAULT_MODELS = ["gemma-3-27b-it", "gemma-3-12b-it", "gemini-2.5-flash", "gemini-2.5-pro"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.full.yaml")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--scale", type=float, default=None,
                    help="override response_scale (e.g. 0.01 for a cheap run)")
    ap.add_argument("--no-skip", action="store_true",
                    help="re-run conditions even if output exists")
    args = ap.parse_args()

    cfg = EvalConfig.load(args.config)
    if args.scale is not None:
        cfg.response_scale = args.scale
        for c in cfg.conditions:
            c.n_responses = max(c.num_turns, int(round(c.n_responses * args.scale)))
    run_eval(args.models, cfg, skip_existing=not args.no_skip)


if __name__ == "__main__":
    main()
