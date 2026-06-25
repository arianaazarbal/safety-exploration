#!/usr/bin/env python
"""Open-ended emotion elicitation (Petri-style) for a target model (Section 4.1).

Example
-------
python scripts/run_petri.py --model gemma-3-27b-it --out outputs/petri/gemma.jsonl
python scripts/run_petri.py --adapter outputs/models/gemma-dpo --key gemma-dpo \
    --out outputs/petri/gemma-dpo.jsonl
"""
from __future__ import annotations

import argparse

import _common  # noqa: F401

from instability.config import TARGET_MODELS, with_adapter
from instability.models.registry import load_model
from instability.petri import run_petri_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, choices=list(TARGET_MODELS))
    ap.add_argument("--adapter", default=None, help="LoRA adapter path (overrides --model)")
    ap.add_argument("--key", default=None, help="output key when using --adapter")
    ap.add_argument("--base-key", default="gemma-3-27b-it-local")
    ap.add_argument("--out", required=True)
    ap.add_argument("--transcripts-per-dim", type=int, default=5)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.adapter:
        spec = with_adapter(args.base_key, args.adapter, args.key or "adapter",
                            args.key or "adapter")
    else:
        spec = TARGET_MODELS[args.model]

    model = load_model(spec)
    run_petri_eval(
        spec, args.out, target_model=model,
        n_transcripts_per_dim=args.transcripts_per_dim,
        n_turns=args.turns, seed=args.seed,
    )


if __name__ == "__main__":
    main()
