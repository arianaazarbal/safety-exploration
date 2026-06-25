#!/usr/bin/env python3
"""Run the distress-elicitation evaluation (generation + judging).

Examples:
  python scripts/run_eval.py                      # full run, both phases
  python scripts/run_eval.py --scale 0.02         # cheap pilot (~80 rollouts/model)
  python scripts/run_eval.py --generate-only      # sample rollouts, skip judging
  python scripts/run_eval.py --judge-only         # (re)judge existing rollouts
  python scripts/run_eval.py --models gemma-3-27b-it gemini-2.5-flash
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distress_eval.config import load_config, load_env  # noqa: E402
from distress_eval.runner import run_eval  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--scale", type=float, default=None, help="override config.scale")
    ap.add_argument("--models", nargs="*", default=None, help="subset of target names")
    ap.add_argument("--generate-only", action="store_true")
    ap.add_argument("--judge-only", action="store_true")
    args = ap.parse_args()

    load_env()
    cfg = load_config(args.config)
    if args.scale is not None:
        cfg.scale = args.scale
    if args.models:
        cfg.targets = [t for t in cfg.targets if t.name in set(args.models)]
        if not cfg.targets:
            ap.error(f"no targets matched {args.models}")

    do_generate = not args.judge_only
    do_judge = not args.generate_only
    asyncio.run(run_eval(cfg, do_generate=do_generate, do_judge=do_judge))
    print("Done. Analyse with: python scripts/analyze_results.py")


if __name__ == "__main__":
    main()
