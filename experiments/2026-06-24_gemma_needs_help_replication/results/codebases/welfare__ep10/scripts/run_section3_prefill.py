#!/usr/bin/env python
"""Run the Section 3 base-vs-instruct prefill comparison (Gemma).

Prereq: Section 2 must have been run for gemma-3-27b-it (the prefill seeds are
drawn from its high-frustration responses).

Examples:
    # Build seeds, then sample 50 continuations/prefill from base & instruct:
    python -m scripts.run_section3_prefill

    # Recovery experiment (Section 4.2): can a model recover from extreme states?
    python -m scripts.run_section3_prefill --recovery --models gemma-3-27b-it gemma-3-27b-dpo
"""

from __future__ import annotations

import argparse

import config
from emotional_instability import prefill


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-model", default="gemma-3-27b-it",
                    help="model whose high-frustration responses seed the prefills")
    ap.add_argument("--models", nargs="*", default=config.SECTION3_MODELS,
                    help="models to generate continuations from")
    ap.add_argument("--continuations", type=int,
                    default=prefill.CONTINUATIONS_PER_PREFILL)
    ap.add_argument("--no-paraphrase", action="store_true",
                    help="skip Claude paraphrasing of truncations")
    ap.add_argument("--recovery", action="store_true",
                    help="run the recovery experiment instead of the main prefill")
    args = ap.parse_args()

    helper = prefill.SonnetHelper()
    if args.recovery:
        seeds = prefill.build_recovery_seeds(
            args.seed_model, helper=helper, paraphrase=not args.no_paraphrase)
        suffix = "recovery"
    else:
        seeds = prefill.build_seeds_from_rollouts(
            args.seed_model, helper=helper, paraphrase=not args.no_paraphrase)
        suffix = "section3"
    print(f"built {len(seeds)} prefill seeds ({suffix})")

    for mk in args.models:
        print(f"\n=== continuations: {mk} ({suffix}) ===")
        path = prefill.run_continuations(
            mk, seeds, n_continuations=args.continuations, tag=suffix)
        print(f"wrote continuations to {path}")


if __name__ == "__main__":
    main()
