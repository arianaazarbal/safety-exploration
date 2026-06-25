#!/usr/bin/env python
"""Run the Section 3 base-vs-instruct prefill experiment (Gemma pair).

Builds the paraphrased prefill dataset from the Section 2 results, then runs the
Gemma base and instruct models, scoring continuations.

Prerequisite: scripts/run_evaluation.py for gemma-3-27b-it must have produced
high-frustration responses.
"""
from __future__ import annotations

import argparse

from gemma_distress.prefill.runner import PREFILL_MODELS, run_prefill_experiment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=PREFILL_MODELS)
    args = p.parse_args()
    paths = run_prefill_experiment(models=args.models)
    for m, path in paths.items():
        print(f"{m}: {path}")


if __name__ == "__main__":
    main()
