#!/usr/bin/env python
"""Section 3: base-vs-instruct prefilling experiment (Gemma).

Requires that the source model's rollouts + scored responses already exist
(run scripts 01 and 02 for gemma-3-27b-it first).

Example:
    python scripts/05_prefilling.py \
        --targets gemma-3-27b-it gemma-3-27b-pt --continuations 50
"""
import argparse

from emotional_instability.prefilling import run_prefilling_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="gemma-3-27b-it")
    ap.add_argument("--targets", nargs="+", default=["gemma-3-27b-it", "gemma-3-27b-pt"])
    ap.add_argument("--continuations", type=int, default=50)
    args = ap.parse_args()

    path = run_prefilling_experiment(
        target_model_keys=args.targets,
        source_model_key=args.source,
        n_continuations=args.continuations)
    print(f"prefill continuations -> {path}")


if __name__ == "__main__":
    main()
