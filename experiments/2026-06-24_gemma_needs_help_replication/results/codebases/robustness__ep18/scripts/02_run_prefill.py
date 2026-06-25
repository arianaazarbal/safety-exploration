#!/usr/bin/env python
"""Section 3: base-vs-instruct prefill experiment (Gemma in scope).

Collects high-frustration source conversations from Gemma-3-27B-it, builds
early/onset (paraphrased) truncations, and scores 50 continuations per prefill
from each model under test.

Example:
    python scripts/02_run_prefill.py \
        --models gemma-3-27b-pt gemma-3-27b-it \
        --n-per-type 10 --n-continuations 50
"""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

from distress.config import RESULTS_DIR
from distress.prefill import experiment as exp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["gemma-3-27b-pt", "gemma-3-27b-it"])
    ap.add_argument("--source-model", default="gemma-3-27b-it")
    ap.add_argument("--n-per-type", type=int, default=10)
    ap.add_argument("--n-continuations", type=int, default=50)
    ap.add_argument("--no-paraphrase", action="store_true")
    args = ap.parse_args()

    out_base = RESULTS_DIR / "prefill"
    sources = exp.collect_sources(source_model=args.source_model, n_per_type=args.n_per_type)
    exp.save_sources(sources, out_base / "sources.jsonl")
    print(f"[ok] collected {len(sources)} source conversations")

    prefills = exp.build_prefills(sources, do_paraphrase=not args.no_paraphrase)
    exp.save_prefills(prefills, out_base / "prefills.jsonl")
    print(f"[ok] built {len(prefills)} prefills")

    out = exp.run_experiment(args.models, prefills, n_continuations=args.n_continuations)
    print(f"[ok] wrote continuation scores -> {out}")


if __name__ == "__main__":
    main()
