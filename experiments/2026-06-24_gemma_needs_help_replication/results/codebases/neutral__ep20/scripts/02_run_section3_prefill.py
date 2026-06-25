#!/usr/bin/env python
"""Section 3: base-vs-instruct divergence via prefilled continuations.

Steps:
  1. Build prefill seeds from the Section 2 instruct results (onset labelling +
     paraphrase via Claude).
  2. Generate + score 50 continuations per seed per model (Gemma base + instruct).
  3. Aggregate.

Prereq: Section 2 must have been run for gemma-3-27b-it.
Requires ANTHROPIC_API_KEY (onset/paraphrase/judge) and GPU/HF access (Gemma).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import config
from gemma_distress.prefill import onset, run_prefill


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=config.SECTION3_MODELS)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip-seeds", action="store_true")
    args = ap.parse_args()

    if not args.skip_seeds:
        onset.build_seeds()
    run_prefill.run_all(args.models, recovery=False, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
