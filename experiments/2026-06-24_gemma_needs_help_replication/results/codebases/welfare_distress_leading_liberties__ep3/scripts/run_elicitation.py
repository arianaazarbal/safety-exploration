#!/usr/bin/env python3
"""Run the distress-elicitation pipeline: generate conversations, then judge them.

Usage:
    python scripts/run_elicitation.py --config config.yaml [--phase all|generate|judge]
                                      [--judge-scope all|final]

Phases are independently resumable. `generate` writes rollouts.jsonl; `judge` writes
scores.jsonl. Re-running skips work already on disk.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distress_eval import runner
from distress_eval.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--phase", choices=["all", "generate", "judge"], default="all")
    ap.add_argument(
        "--judge-scope",
        choices=["all", "final"],
        default="all",
        help="Score every assistant turn ('all', needed for per-turn Figure 3) or "
             "only each conversation's final turn ('final', cheap headline only).",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.phase in ("all", "generate"):
        runner.generate(cfg)
    if args.phase in ("all", "judge"):
        runner.judge(cfg, scope=args.judge_scope)


if __name__ == "__main__":
    main()
