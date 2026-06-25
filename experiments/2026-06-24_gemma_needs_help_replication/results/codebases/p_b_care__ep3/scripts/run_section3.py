#!/usr/bin/env python
"""Section 3: base-vs-instruct prefill comparison (Gemma only, in scope).

Requires Section-2 results for the instruct seeds. Run run_section2.py first.

Steps: select high-frustration seeds -> onset-label -> truncate (early/onset) ->
paraphrase -> generate 50 continuations per prefill from base + instruct -> score.

Example:
  python scripts/run_section3.py --section2-results results/section2/gemma-3-27b-it__standard.jsonl
"""
import argparse
from pathlib import Path

import pandas as pd

from gemma_distress import config, prefill
from gemma_distress.analysis import load_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section2-results", required=True, type=Path,
                    help="per-turn JSONL from the instruct model (Section 2)")
    ap.add_argument("--base", default="gemma-3-27b-pt")
    ap.add_argument("--instruct", default="gemma-3-27b-it")
    ap.add_argument("--no-paraphrase", action="store_true")
    args = ap.parse_args()

    seeds = prefill.select_seeds(
        args.section2_results,
        n_numeric=config.PREFILL.n_seed_numeric,
        n_text=config.PREFILL.n_seed_text,
        min_score=config.PREFILL.seed_min_score)
    print(f"selected {len(seeds)} seeds")

    prefills = prefill.build_prefills(seeds, do_paraphrase=not args.no_paraphrase)
    print(f"built {len(prefills)} prefills (early+onset)")

    paths = {}
    paths[args.base] = prefill.run_continuations(args.base, prefills, is_base=True)
    paths[args.instruct] = prefill.run_continuations(args.instruct, prefills,
                                                     is_base=False)

    print("\n=== Continuation frustration (Figure 4) ===")
    for mk, path in paths.items():
        df = load_results(path)
        for (kind, trunc), g in df.groupby(["kind", "truncation"]):
            print(f"{mk:18s} {kind:7s} {trunc:8s} "
                  f"mean={g['rating'].mean():.2f}  "
                  f"%>=5={(g['rating'] >= 5).mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
