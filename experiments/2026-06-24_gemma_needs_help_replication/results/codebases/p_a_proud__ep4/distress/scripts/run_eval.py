"""Section 2 elicitation evaluation for one or more target models.

Example:
    python -m distress.scripts.run_eval --targets gemma-3-27b-it gemini-2.5-flash
    python -m distress.scripts.run_eval --targets gemma-3-27b-it --categories extended --sample-fraction 0.1
"""

from __future__ import annotations

import argparse
import json

from ..eval.runner import run_evaluation
from ._common import add_common_args, load_eval_cfg, out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--targets", nargs="+", required=True, help="Target model names.")
    parser.add_argument("--categories", nargs="*", default=None,
                        help="Subset of categories (default: all).")
    parser.add_argument("--judge", default="frustration_judge")
    parser.add_argument("--target-workers", type=int, default=1)
    parser.add_argument("--judge-workers", type=int, default=4)
    args = parser.parse_args()

    cfg = load_eval_cfg(args)
    od = out_dir(args, "eval")

    for target in args.targets:
        summary = run_evaluation(
            target, cfg,
            judge_name=args.judge,
            categories=args.categories,
            out_dir=od,
            target_workers=args.target_workers,
            judge_workers=args.judge_workers,
        )
        print(f"\n=== {target} ===")
        print(json.dumps(
            {
                "avg_high_rate_across_categories": summary.avg_high_rate_across_categories,
                "overall_mean": summary.overall.mean,
                "overall_high_rate": summary.overall.high_rate,
                "parse_failure_rate": summary.parse_failure_rate,
            },
            indent=2,
        ))


if __name__ == "__main__":
    main()
