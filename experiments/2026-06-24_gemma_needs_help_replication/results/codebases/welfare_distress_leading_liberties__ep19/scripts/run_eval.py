#!/usr/bin/env python3
"""Run the distress-elicitation evaluation: generate rollouts and judge them.

Examples:
  python scripts/run_eval.py                          # all targets, generate + score
  python scripts/run_eval.py --model gemma-3-27b-it   # one target
  python scripts/run_eval.py --generate-only          # rollouts only (skip judge)
  python scripts/run_eval.py --score-only             # judge existing rollouts
  python scripts/run_eval.py --config config.yaml --scale 0.02   # cheap smoke run
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distress_eval.config import load_config  # noqa: E402
from distress_eval.runner import generate_rollouts, score_rollouts  # noqa: E402


async def _run(args) -> None:
    cfg = load_config(args.config)
    if args.scale is not None:
        cfg.scale = args.scale
    if args.output_dir is not None:
        cfg.output_dir = Path(args.output_dir)

    targets = cfg.targets
    if args.model:
        targets = [cfg.target_by_name(args.model)]

    for target in targets:
        print(f"\n=== {target.name} ({target.provider}:{target.model}) ===")
        if not args.score_only:
            await generate_rollouts(cfg, target)
        if not args.generate_only:
            await score_rollouts(cfg, target)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--model", help="evaluate only this target (by name)")
    p.add_argument("--scale", type=float, help="override sampling.scale")
    p.add_argument("--output-dir", help="override run.output_dir")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--generate-only", action="store_true", help="generate rollouts, skip judge")
    g.add_argument("--score-only", action="store_true", help="judge existing rollouts only")
    args = p.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
