#!/usr/bin/env python3
"""Section 3: prefill experiment (Gemma base vs instruct).

Requires a prior Section 2 run for gemma-3-27b-it (it sources high-frustration
conversations from results/eval_gemma-3-27b-it.jsonl + rollouts file).

  python scripts/run_prefill.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESULTS_DIR
from src.prefill.run_prefill import run_prefill_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-file", default=str(RESULTS_DIR / "eval_gemma-3-27b-it.jsonl"))
    args = ap.parse_args()
    path = run_prefill_experiment(Path(args.eval_file))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
