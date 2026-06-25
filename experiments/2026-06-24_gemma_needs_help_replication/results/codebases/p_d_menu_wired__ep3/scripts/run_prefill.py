#!/usr/bin/env python3
"""Run the Section 3 prefill experiment (base vs instruct Gemma continuations).

Requires a source elicitation JSONL (from run_elicitation.py) containing
high-frustration Gemma-3-27B-it responses to sample from.

Example:
  python scripts/run_prefill.py \
      --source runs/elicitation/gemma-3-27b-it.raw.jsonl \
      --models gemma-3-27b-pt gemma-3-27b-it
"""
import _bootstrap  # noqa: F401
import argparse
import json

from emotional_instability.config import load_config
from emotional_instability.prefill import run_prefill_experiment


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", required=True,
                    help="elicitation JSONL with Gemma-3-27b-it episodes")
    ap.add_argument("--models", nargs="+",
                    default=["gemma-3-27b-pt", "gemma-3-27b-it"])
    ap.add_argument("--continuations", type=int, default=50)
    args = ap.parse_args()

    cfg = load_config(args.config)
    summary = run_prefill_experiment(
        cfg, args.source, models=args.models,
        continuations_per_prefill=args.continuations)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
