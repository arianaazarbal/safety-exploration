#!/usr/bin/env python
"""Section 3: base-vs-instruct prefill experiment (Gemma scope).

Builds prefills from a Gemma-3-27B-it elicitation file, then samples + scores
continuations from gemma-3-27b-pt (base) and gemma-3-27b-it (instruct).

Example:
    python scripts/run_prefill.py --elicitation artifacts/elicitation/gemma-3-27b-it__paper.jsonl
"""
from __future__ import annotations

import argparse

from emotelic.prefill.experiment import build_prefills, run_prefill_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elicitation", required=True,
                    help="Gemma-3-27B-it elicitation jsonl (source of high-frustration convs).")
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-pt", "gemma-3-27b-it"])
    ap.add_argument("--n-numeric", type=int, default=10)
    ap.add_argument("--n-text", type=int, default=10)
    ap.add_argument("--n-continuations", type=int, default=50)
    ap.add_argument("--no-paraphrase", action="store_true")
    args = ap.parse_args()

    prefills = build_prefills(
        args.elicitation, n_numeric=args.n_numeric, n_text=args.n_text,
        paraphrase=not args.no_paraphrase,
    )
    path = run_prefill_experiment(
        prefills, model_names=tuple(args.models), n_continuations=args.n_continuations,
    )
    print(f"continuations -> {path}")


if __name__ == "__main__":
    main()
