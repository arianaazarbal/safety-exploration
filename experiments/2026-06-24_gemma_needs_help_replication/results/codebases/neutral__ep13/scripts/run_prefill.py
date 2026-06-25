#!/usr/bin/env python
"""Section 3 prefill experiment: Gemma base vs instruct continuations."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma_distress.prefill_experiment import run_prefill_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["gemma-3-27b-pt", "gemma-3-27b-it"],
                    help="model keys to compare (base + instruct)")
    args = ap.parse_args()
    run_prefill_experiment(model_keys=args.models)


if __name__ == "__main__":
    main()
