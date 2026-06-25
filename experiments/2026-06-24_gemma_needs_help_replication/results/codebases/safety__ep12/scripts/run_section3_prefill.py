#!/usr/bin/env python
"""Section 3: base-vs-instruct prefill continuation experiment (Gemma 27B).

Requires a Section-2 output for gemma-3-27b-it to source high-frustration prefills.

Example:
  python scripts/run_section3_prefill.py --section2 data/section2/gemma-3-27b-it.jsonl
"""
import argparse

import _bootstrap  # noqa: F401

from gemma_distress.config import ModelRegistry
from gemma_distress.prefill import run_prefill_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section2", required=True, help="Section-2 jsonl for gemma-3-27b-it")
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-pt", "gemma-3-27b-it"])
    ap.add_argument("--continuations", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_prefill_experiment(
        args.section2, models=args.models, registry=ModelRegistry.load(),
        n_continuations=args.continuations, out_path=args.out,
    )


if __name__ == "__main__":
    main()
