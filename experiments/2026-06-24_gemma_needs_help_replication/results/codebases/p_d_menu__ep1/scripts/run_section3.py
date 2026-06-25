#!/usr/bin/env python3
"""Run the Section 3 prefill base-vs-instruct experiment (Gemma only).

Steps: collect high-frustration sources from Gemma-27B-it -> label onset ->
truncate (early/onset) -> paraphrase -> generate continuations for
gemma-3-27b-pt and gemma-3-27b-it -> score and summarise.
"""
from __future__ import annotations

import argparse

from _common import add_common_args, get_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+",
                        default=["gemma-3-27b-pt", "gemma-3-27b-it"])
    parser.add_argument("--n-continuations", type=int, default=50)
    add_common_args(parser)
    args = parser.parse_args()

    cfg = get_config(args)
    out_dir = args.out or "outputs/section3"

    from emotional_instability.prefill.continuations import (
        build_prefill_items, collect_source_conversations, run_continuations,
    )

    print("Collecting high-frustration source conversations from Gemma-27B-it...")
    sources = collect_source_conversations(cfg, out_dir)
    print(f"  collected {len(sources)} sources")

    print("Labelling onset + truncating + paraphrasing...")
    items = build_prefill_items(cfg, sources)
    print(f"  built {len(items)} prefill items")

    print("Generating + scoring continuations...")
    reports = run_continuations(cfg, items, out_dir, models=args.models,
                                n_continuations=args.n_continuations)
    for model, rep in reports.items():
        print(f"  {model}:")
        for cond, summ in rep.by_condition.items():
            print(f"    {cond:14s} mean={summ['mean']:.2f} %>=5={summ['pct_high']:.1f}%")


if __name__ == "__main__":
    main()
