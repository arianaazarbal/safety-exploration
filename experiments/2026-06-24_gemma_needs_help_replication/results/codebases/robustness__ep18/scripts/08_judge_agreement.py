#!/usr/bin/env python
"""Section 2.1: judge-reliability cross-check (Claude vs GPT-5-mini).

Example:
    python scripts/08_judge_agreement.py \
        --models gemma-3-27b-it gemini-2.5-flash --n 260
"""
import _bootstrap  # noqa: F401
import argparse
import json

from distress.eval.agreement import run_agreement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--secondary", default="gpt-5-mini")
    ap.add_argument("--n", type=int, default=260)
    args = ap.parse_args()

    result = run_agreement(args.models, secondary_judge=args.secondary, n_sample=args.n)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
