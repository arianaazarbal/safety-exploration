#!/usr/bin/env python
"""Section 3: base-vs-instruct prefill continuation experiment (Gemma only).

Requires a Section-2 scores file for the source model (gemma-3-27b-it) so the
high-frustration source conversations can be selected. Build prefills (onset +
early truncations, paraphrased), then have base & instruct Gemma each generate
50 continuations per prefill.

    python scripts/run_section3.py \
        --source-scores outputs/scores/gemma-3-27b-it.jsonl \
        --models gemma-3-27b-pt gemma-3-27b-it
"""
from __future__ import annotations

import argparse

from gemma_distress.prefill.build_prefills import build_prefills_from_rollouts
from gemma_distress.prefill.run_prefill import (build_recovery_prefills,
                                                run_prefill_experiment,
                                                summarize_prefill)
from gemma_distress.utils.io import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-scores", required=True,
                    help="Section-2 scores JSONL for the source model")
    ap.add_argument("--source-model", default="gemma-3-27b-it")
    ap.add_argument("--models", nargs="+",
                    default=["gemma-3-27b-pt", "gemma-3-27b-it"])
    ap.add_argument("--recovery", action="store_true",
                    help="run the Section 4.2 recovery test instead")
    ap.add_argument("--no-paraphrase", action="store_true")
    args = ap.parse_args()

    records = list(read_jsonl(args.source_scores))
    if args.recovery:
        prefills = build_recovery_prefills(
            records, source_model=args.source_model,
            do_paraphrase=not args.no_paraphrase)
        tag = "recovery"
    else:
        prefills = build_prefills_from_rollouts(
            records, source_model=args.source_model,
            do_paraphrase=not args.no_paraphrase)
        tag = "prefill"
    print(f"Built {len(prefills)} prefills.")

    path = run_prefill_experiment(args.models, prefills, tag=tag)
    print(f"\n=== {tag} continuation summary ===")
    for k, v in sorted(summarize_prefill(path).items()):
        print(f"{k:50s} mean={v['mean']:.2f} %>=5={100*v['pct_high']:.1f}% "
              f"n={v['n']}")


if __name__ == "__main__":
    main()
