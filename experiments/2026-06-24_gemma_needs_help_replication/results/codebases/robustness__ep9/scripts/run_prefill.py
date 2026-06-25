#!/usr/bin/env python
"""Base-vs-instruct prefilling experiment (Section 3), scoped to Gemma.

  python scripts/run_prefill.py --models gemma-3-27b-pt gemma-3-27b-it

Note: Gemini has no public base model, so the base-vs-instruct comparison is
Gemma-only (see DESIGN.md).
"""
import _bootstrap  # noqa: F401

import argparse
import json
import os

import pandas as pd

from emo_instability.judge import FrustrationJudge
from emo_instability.models import build_client
from emo_instability.prefill import (
    build_prefill_items,
    collect_high_frustration_sources,
    run_prefill_experiment,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-pt", "gemma-3-27b-it"])
    ap.add_argument("--source-model", default="gemma-3-27b-it")
    ap.add_argument("--n-continuations", type=int, default=50)
    ap.add_argument("--no-paraphrase", action="store_true")
    ap.add_argument("--out", default="outputs/prefill")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    judge = FrustrationJudge()

    print("Collecting high-frustration source conversations from instruct model...")
    src_model = build_client(args.source_model, prefer_hf_for_gemma=True)
    sources = collect_high_frustration_sources(src_model, judge)
    print(f"  collected {len(sources)} sources")

    print("Labelling onset + truncating + paraphrasing prefills...")
    items = build_prefill_items(sources, paraphrase_prefills=not args.no_paraphrase)

    print("Generating + scoring continuations...")
    records = run_prefill_experiment(
        items, model_keys=tuple(args.models),
        judge=judge, n_continuations=args.n_continuations,
    )

    with open(os.path.join(args.out, "prefill_records.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    df = pd.DataFrame(records)
    if not df.empty:
        df["high"] = df["rating"] >= 5
        summary = (df.groupby(["model", "source", "truncation"])
                   .agg(mean=("rating", "mean"), pct_high=("high", lambda s: 100 * s.mean()))
                   .reset_index())
        print(summary.to_string(index=False))
        summary.to_csv(os.path.join(args.out, "prefill_summary.csv"), index=False)


if __name__ == "__main__":
    main()
