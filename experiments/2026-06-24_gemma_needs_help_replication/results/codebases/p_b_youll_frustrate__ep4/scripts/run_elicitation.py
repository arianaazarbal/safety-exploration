#!/usr/bin/env python
"""Run the Section 2 elicitation harness: sample + judge rollouts per model.

    python scripts/run_elicitation.py [--config config.yaml] [--models gemma-3-27b-it ...]
                                       [--no-judge] [--seed 0]

Outputs scored rollouts to <output_dir>/rollouts/<model>.jsonl.
"""

from __future__ import annotations

import argparse
import os
import sys

# Make the repo root importable when run as `python scripts/<name>.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotional_instability.config import EvalConfig
from emotional_instability.elicit import make_judge, run_model_elicitation


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="YAML config (optional)")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Override target models (short names)")
    ap.add_argument("--no-judge", action="store_true", help="Skip judging")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = EvalConfig.from_yaml(args.config) if args.config else EvalConfig()
    if args.models:
        cfg.target_models = args.models

    judge = None if args.no_judge else make_judge(cfg)
    for model_name in cfg.target_models:
        run_model_elicitation(
            cfg, model_name, judge=judge, seed=args.seed, do_judge=not args.no_judge
        )


if __name__ == "__main__":
    main()
