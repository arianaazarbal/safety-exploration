#!/usr/bin/env python3
"""Judge-reliability validation (Section 2.1): Claude-Sonnet vs GPT-5-mini.

Reports Pearson r and % within one point over a random subset of responses.

Example:
    python scripts/run_judge_agreement.py runs/elicitation/*.jsonl --n 260
"""

import argparse
import glob
import json

import _bootstrap  # noqa: F401
from emotional_instability.config import load_config
from emotional_instability.eval.agreement import run_agreement


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="elicitation result JSONL files/globs")
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=260)
    args = ap.parse_args()

    paths = [p for pat in args.results for p in glob.glob(pat)]
    cfg = load_config(args.config)
    result = run_agreement(cfg, paths, n=args.n)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
