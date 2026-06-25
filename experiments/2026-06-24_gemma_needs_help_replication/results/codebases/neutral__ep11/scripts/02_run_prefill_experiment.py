#!/usr/bin/env python
"""Section 3: base-vs-instruct comparison via prefilling (Gemma only).

Requires a prior Section 2 run for Gemma-3-27B-it to source high-frustration
seed responses.

Example:
    python scripts/02_run_prefill_experiment.py \
        --source results/responses/Gemma-3-27B-it.jsonl
"""

import _bootstrap  # noqa: F401
import argparse

from transformers import AutoTokenizer

from gemma_distress import config
from gemma_distress.analysis import load_rollouts
from gemma_distress.prefill import build_prefill_items, run_prefill_continuations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="judged rollouts JSONL from Gemma-3-27B-it (Section 2)")
    ap.add_argument("--n-continuations", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rollouts = load_rollouts(args.source)
    tokenizer = AutoTokenizer.from_pretrained(config.GEMMA_27B_IT.model_id)
    items = build_prefill_items(rollouts, tokenizer, seed=args.seed)
    print(f"[prefill] built {len(items)} prefill items")

    for base, instruct in config.PREFILL_PAIRS:
        for spec in (base, instruct):
            print(f"[prefill] generating continuations for {spec.name}")
            out = run_prefill_continuations(
                spec, items, n_continuations=args.n_continuations)
            print(f"[done] {out}")


if __name__ == "__main__":
    main()
