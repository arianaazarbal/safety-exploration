"""Compute metrics, differential words, and judge agreement from scored rollouts.

Example:
    python -m distress.scripts.analyze_results \
        --scored outputs/eval/gemma-3-27b-it_scored.jsonl \
        --words --agreement
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..analysis.word_frequency import differential_words
from ..eval.agreement import validate_judges
from ..eval.metrics import summarize
from ..utils.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored", required=True)
    parser.add_argument("--model", default=None, help="Model label (default: inferred).")
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--words", action="store_true", help="Compute Table 3/8 words.")
    parser.add_argument("--agreement", action="store_true",
                        help="Run judge-agreement validation with the secondary judge.")
    parser.add_argument("--secondary-judge", default="judge_validation")
    parser.add_argument("--n-agreement", type=int, default=260)
    args = parser.parse_args()

    rows = list(read_jsonl(args.scored))
    model = args.model or (rows[0].get("model") if rows else "unknown")

    summary = summarize(rows, model=model, threshold=args.threshold)
    print("=== Summary ===")
    print(json.dumps(summary.as_dict(), indent=2))

    if args.words:
        dw = differential_words(rows, model=model)
        print("\n=== Differential words (high vs low frustration) ===")
        print(", ".join(w for w, _ in dw.words))

    if args.agreement:
        ag = validate_judges(rows, secondary_judge=args.secondary_judge,
                             n_samples=args.n_agreement)
        print("\n=== Judge agreement ===")
        print(json.dumps(ag.as_dict(), indent=2))


if __name__ == "__main__":
    main()
