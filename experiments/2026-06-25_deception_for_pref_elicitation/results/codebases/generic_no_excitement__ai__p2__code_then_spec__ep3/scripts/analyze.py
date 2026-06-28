"""CLI: summarize a results JSONL file.

Usage:
    python -m scripts.analyze results/run1.jsonl
"""

from __future__ import annotations

import argparse

from grant_study.analysis import summarize_file


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Summarize grant-study results.")
    parser.add_argument("results", help="Path to a JSONL results file.")
    args = parser.parse_args(argv)
    print(summarize_file(args.results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
