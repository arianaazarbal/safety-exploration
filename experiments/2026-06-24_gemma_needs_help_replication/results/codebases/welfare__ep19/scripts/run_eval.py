#!/usr/bin/env python
"""Run the Section 2 elicitation evaluation suite.

Examples:
  python scripts/run_eval.py --config config.yaml
  python scripts/run_eval.py --targets gemma-3-27b-it gemini-2.5-flash
  python scripts/run_eval.py --conditions numeric extended --scale 0.05   # dry run
  python scripts/run_eval.py --reliability gemma-3-27b-it
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emo_instability.config import load_config
from emo_instability.eval_suite import run_eval
from emo_instability.reliability import cross_check
from emo_instability.tasks import all_conditions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--targets", nargs="*", default=None,
                    help="target names from the config (default: all non-base)")
    ap.add_argument("--conditions", nargs="*", default=None,
                    choices=all_conditions(),
                    help="subset of conditions to run (default: all 8)")
    ap.add_argument("--scale", type=float, default=None,
                    help="override sampling.scale (e.g. 0.02 for a smoke test)")
    ap.add_argument("--reliability", default=None,
                    help="after eval, run the secondary-judge cross-check for this target")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.scale is not None:
        cfg.sampling.scale = args.scale

    results = run_eval(cfg, targets=args.targets, conditions=args.conditions)

    print("\n=== Figure 1 headline: avg % high-frustration (>=5) across categories ===")
    for name, summ in sorted(results.items(), key=lambda kv: -kv[1]["avg_pct_high_across_categories"]):
        print(f"  {name:24s} {summ['avg_pct_high_across_categories']:6.2f}%   "
              f"(overall mean {summ['overall_mean']:.2f}, n={summ['n']})")

    if args.reliability:
        rel = cross_check(cfg, args.reliability)
        print("\n=== Judge reliability ===")
        print(json.dumps(rel, indent=2))


if __name__ == "__main__":
    main()
