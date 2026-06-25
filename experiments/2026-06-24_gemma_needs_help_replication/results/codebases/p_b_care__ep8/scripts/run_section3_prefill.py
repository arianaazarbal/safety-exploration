#!/usr/bin/env python
"""Section 3: base-vs-instruct prefilling experiment (Gemma only).

Requires a completed Section 2 run for gemma-3-27b-it (to source high-frustration
seeds). Writes prefills + scored continuations to results/section3/.
"""
import _bootstrap  # noqa: F401
import config
from src.prefill import run_prefill_experiment


def main():
    run_prefill_experiment()
    print(f"Done. Results in {config.RESULTS_DIR / 'section3'}")


if __name__ == "__main__":
    main()
