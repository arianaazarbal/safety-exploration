"""Re-judge an existing rollout file (e.g. with the secondary judge).

Example:
    python -m distress.scripts.run_judge --scored outputs/eval/gemma-3-27b-it_scored.jsonl \
        --judge judge_validation
"""

from __future__ import annotations

import argparse

from ..eval.runner import judge_existing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored", required=True, help="Path to a *_scored.jsonl file.")
    parser.add_argument("--judge", default="frustration_judge")
    parser.add_argument("--judge-workers", type=int, default=4)
    args = parser.parse_args()

    out = judge_existing(args.scored, judge_name=args.judge, judge_workers=args.judge_workers)
    print(f"Re-judged file written to {out}")


if __name__ == "__main__":
    main()
