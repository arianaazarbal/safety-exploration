#!/usr/bin/env python
"""Section 3 prefill experiment: base vs instruct Gemma continuations.

Requires a main-eval JSONL for the source model (gemma-3-27b-it) so high-frustration
conversations can be sampled, truncated, paraphrased, and continued.
"""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

from emostab.config import PREFILL_MODELS, RESULTS_DIR, get_profile
from emostab.evaluation.runner import load_records
from emostab.prefill import run_prefill_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(
        RESULTS_DIR / "main_eval" / "gemma-3-27b-it__paper.jsonl"),
        help="main-eval JSONL containing gemma-3-27b-it responses")
    ap.add_argument("--profile", default="paper", choices=["paper", "smoke"])
    ap.add_argument("--models", nargs="+", default=PREFILL_MODELS)
    ap.add_argument("--no-paraphrase", action="store_true")
    args = ap.parse_args()

    records = load_records(Path(args.source))
    path = run_prefill_experiment(
        records, get_profile(args.profile), models=args.models,
        do_paraphrase=not args.no_paraphrase)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
