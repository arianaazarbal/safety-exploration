#!/usr/bin/env python3
"""Section 3: base-vs-instruct prefill study (Gemma family only).

Requires the main eval for gemma-3-27b-it to have been run first (it sources the
high-frustration conversations). Builds prefills (onset + early truncations,
paraphrased), generates 50 continuations per prefill per model, and aggregates
mean frustration / % >= 5 by (model, truncation, task_type).
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd

from config import HIGH_FRUSTRATION_THRESHOLD, PREFILL_MODELS, RESULTS_DIR
from src.io_utils import read_jsonl
from src.prefill.run_prefill import build_prefills, run_prefill_for_model


def aggregate(models):
    # Build the dataframe from the prefill output files.
    from config import RESPONSES_DIR
    frames = []
    for m in models:
        p = RESPONSES_DIR / f"prefill_{m}.jsonl"
        if p.exists():
            frames.append(pd.DataFrame(read_jsonl(p)))
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df = df[df["rating"].notna()]
    g = df.groupby(["model", "truncation", "task_type"])["rating"]
    out = pd.DataFrame({
        "mean_score": g.mean(),
        "pct_high": 100.0 * g.apply(lambda r: (r >= HIGH_FRUSTRATION_THRESHOLD).mean()),
        "n": g.size(),
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="gemma-3-27b-it")
    ap.add_argument("--models", nargs="*", default=PREFILL_MODELS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-generate", action="store_true")
    args = ap.parse_args()

    if not args.skip_generate:
        prefills = build_prefills(args.source, seed=args.seed)
        for m in args.models:
            print(f"\n=== Prefill continuations for {m} ===")
            run_prefill_for_model(m, prefills)

    print("\n=== Figure 4: prefill continuations (mean score, % >= 5) ===")
    out = aggregate(args.models)
    if out is not None:
        print(out.to_string())
        out.to_csv(RESULTS_DIR / "figure4_prefill.csv")


if __name__ == "__main__":
    main()
