#!/usr/bin/env python
"""Section 3: base-vs-instruct prefilling (Gemma within scope).

Requires a Section 2 results file for gemma-3-27b-it (the high-frustration
source responses). Builds prefill cases (onset-label + truncate + paraphrase),
then generates and scores continuations for base + instruct Gemma.
"""
import _bootstrap  # noqa: F401
import argparse
import json

from emotional_instability.config import SECTION3_TARGETS
from emotional_instability.models.registry import build_model
from emotional_instability.prefill.runner import (
    aggregate_section3, build_prefill_cases, run_continuations,
    select_high_frustration_cases,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemma-results", default="results/section2/gemma-3-27b-it.jsonl")
    ap.add_argument("--models", nargs="+", default=SECTION3_TARGETS)
    ap.add_argument("--n-continuations", type=int, default=50)
    ap.add_argument("--no-paraphrase", action="store_true")
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    cases = select_high_frustration_cases(args.gemma_results)
    print(f"Selected {len(cases)} high-frustration source responses.")

    # Use the instruct Gemma tokenizer for the 'early' 20-token truncation.
    tok_model = build_model("gemma-3-27b-it", load_in_4bit=args.load_in_4bit)
    prefill_cases = build_prefill_cases(
        cases, tokenizer_model=tok_model, paraphrase=not args.no_paraphrase
    )
    with open("results/section3/prefill_cases.json", "w") as f:
        json.dump([c.__dict__ for c in prefill_cases], f, indent=2)

    path = run_continuations(
        prefill_cases, model_names=args.models,
        n_continuations=args.n_continuations, load_in_4bit=args.load_in_4bit,
    )
    agg = aggregate_section3(path)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
