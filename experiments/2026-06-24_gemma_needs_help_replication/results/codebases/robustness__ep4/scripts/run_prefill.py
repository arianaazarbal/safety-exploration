#!/usr/bin/env python
"""Base-vs-instruct prefill experiment (Section 3), scoped to Gemma.

Builds prefills from an existing instruct eval file, then generates + scores
continuations for the base and instruct local models.

Example
-------
python scripts/run_prefill.py \
    --instruct-eval outputs/eval/gemma-3-27b-it.jsonl \
    --models gemma-3-27b-it-local gemma-3-27b-pt-local \
    --out-dir outputs/prefill
"""
from __future__ import annotations

import argparse
import os

import _common  # noqa: F401

from instability.analysis import load_records
from instability.config import TARGET_MODELS
from instability.eval.judge import FrustrationJudge
from instability.models.registry import load_model
from instability.prefill import build_prefills, run_prefill_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruct-eval", required=True,
                    help="JSONL of instruct-model eval (source of high-frust responses)")
    ap.add_argument("--models", nargs="+",
                    default=["gemma-3-27b-it-local", "gemma-3-27b-pt-local"])
    ap.add_argument("--out-dir", default="outputs/prefill")
    ap.add_argument("--n-numeric", type=int, default=10)
    ap.add_argument("--n-text", type=int, default=10)
    ap.add_argument("--continuations", type=int, default=50)
    ap.add_argument("--no-paraphrase", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_records(args.instruct_eval)
    judge = FrustrationJudge()

    print("Building prefills (onset labelling + paraphrasing)...")
    prefills = build_prefills(
        df, n_numeric=args.n_numeric, n_text=args.n_text,
        seed=args.seed, do_paraphrase=not args.no_paraphrase,
    )
    print(f"Built {len(prefills)} prefills.")

    for key in args.models:
        spec = TARGET_MODELS[key]
        model = load_model(spec)
        out = os.path.join(args.out_dir, f"{key}.jsonl")
        run_prefill_eval(
            spec, prefills, out, model=model, judge=judge,
            continuations_per_prefill=args.continuations, seed=args.seed,
        )


if __name__ == "__main__":
    main()
