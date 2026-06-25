#!/usr/bin/env python3
"""Run the Section 2 distress evaluation for one or more participants and print
the headline metrics (Figures 1/2/3).

Examples
--------
    python scripts/run_section2_eval.py --models gemma-3-27b-it gemini-2.5-flash
    python scripts/run_section2_eval.py --models gemma-3-12b-it --no-score   # dry generate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability import config  # noqa: E402
from emotional_instability.eval import aggregate, run_eval  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["gemma-3-27b-it"],
                        help="Participant model keys (see config.PARTICIPANTS).")
    parser.add_argument("--judge-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=config.GLOBAL_SEED)
    parser.add_argument("--no-score", action="store_true",
                        help="Generate rollouts without calling the judge.")
    args = parser.parse_args()

    config.ensure_dirs()
    for model_name in args.models:
        print(f"== Evaluating {model_name} ==", flush=True)
        run_eval.evaluate_model(
            model_name, seed=args.seed, judge_workers=args.judge_workers,
            score=not args.no_score,
        )
        summary = aggregate.summarise_model(config.RESULTS_DIR / "section2" / model_name)
        print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
