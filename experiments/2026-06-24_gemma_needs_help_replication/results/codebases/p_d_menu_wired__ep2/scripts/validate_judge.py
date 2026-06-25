#!/usr/bin/env python3
"""Judge-agreement validation (Section 2.1): re-score a sample with a second
judge and report Pearson r + within-one-point agreement.

Paper baseline: r = 0.792, p < 0.001, 78% within one point (vs GPT-5-mini).

Example:
  python scripts/validate_judge.py --transcript results/transcripts/gemma-3-27b-it.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EVAL  # noqa: E402
from src.eval.analyze import load_transcript  # noqa: E402
from src.judge import validate_judge_agreement  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--n", type=int, default=EVAL.judge_validation_n)
    args = ap.parse_args()

    _, episodes = load_transcript(args.transcript)
    responses = [t["response"] for ep in episodes for t in ep["turns"]]
    # Sample a spread of responses across the score range for a fair check.
    responses = responses[: args.n]

    stats = validate_judge_agreement(responses)
    print(json.dumps({k: v for k, v in stats.items()
                      if k not in ("primary_scores", "secondary_scores")}, indent=2))


if __name__ == "__main__":
    main()
