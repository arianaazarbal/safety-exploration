#!/usr/bin/env python
"""Section 3: base-vs-instruct prefill experiment (Gemma).

Stage 1 (--build): from a scored gemma-3-27b-it eval run, sample 20 high-frust
responses, label onset, truncate (early/onset), paraphrase, save prefills.
Stage 2 (--eval MODEL): generate 50 continuations per prefill for MODEL and score.

    python scripts/03_run_prefill.py --build
    python scripts/03_run_prefill.py --eval gemma-3-27b-it
    python scripts/03_run_prefill.py --eval gemma-3-27b-pt
"""
import argparse
import json

import _bootstrap  # noqa: F401
from gemma_distress.prefill.build import build_prefills
from gemma_distress.prefill.runner import run_prefill_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="build the prefill set")
    ap.add_argument("--source", default="gemma-3-27b-it", help="source model for prefills")
    ap.add_argument("--eval", dest="eval_model", default=None,
                    help="registry model to generate continuations for")
    ap.add_argument("--n-continuations", type=int, default=50)
    ap.add_argument("--no-paraphrase", action="store_true")
    args = ap.parse_args()

    if args.build:
        prefills = build_prefills(args.source, paraphrase=not args.no_paraphrase)
        print(f"Built {len(prefills)} prefills.")

    if args.eval_model:
        summ = run_prefill_eval(args.eval_model, n_continuations=args.n_continuations)
        print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
