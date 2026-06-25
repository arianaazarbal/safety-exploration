#!/usr/bin/env python
"""Section 3 base-vs-instruct prefill experiment (and Section 4 recovery test).

Builds prefills from high-frustration Gemma-27B-it responses (requires a prior
Section 2 run with transcripts kept), then samples + scores 50 continuations per
prefill for the base and instruct models.

Examples
--------
python scripts/run_prefill.py --pairs gemma-3-27b-pt gemma-3-27b-it
python scripts/run_prefill.py --recovery --models gemma-3-27b-it
"""

from __future__ import annotations

import argparse
import os

from emotional_instability.config import PATHS, PREFILL_PAIRS
from emotional_instability.models.registry import load_backend
from emotional_instability.prefill.continuations import (
    build_recovery_prefills,
    build_section3_prefills,
    run_continuations,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=None,
                    help="models to generate continuations with (default: both "
                         "halves of every PREFILL_PAIRS entry)")
    ap.add_argument("--n-continuations", type=int, default=50)
    ap.add_argument("--no-paraphrase", action="store_true")
    ap.add_argument("--recovery", action="store_true",
                    help="run the Section 4 recovery experiment instead")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    PATHS.ensure()
    # Tokenizer for token-accurate truncation (use the instruct model's).
    from transformers import AutoTokenizer
    from emotional_instability.config import MODEL_REGISTRY

    tok = AutoTokenizer.from_pretrained(MODEL_REGISTRY["gemma-3-27b-it"].model_id)

    if args.recovery:
        prefills = build_recovery_prefills(
            PATHS.scores, tokenizer=tok, do_paraphrase=not args.no_paraphrase,
            seed=args.seed,
        )
        models = args.models or ["gemma-3-27b-it"]
        tag = "recovery"
    else:
        prefills = build_section3_prefills(
            PATHS.scores, tokenizer=tok, do_paraphrase=not args.no_paraphrase,
            seed=args.seed,
        )
        if args.models:
            models = args.models
        else:
            models = sorted({m for pair in PREFILL_PAIRS for m in pair})
        tag = "section3"

    print(f"[run_prefill] built {len(prefills)} prefills; models={models}")
    for model in models:
        backend = load_backend(model)
        out = os.path.join(PATHS.prefill, f"{tag}__{model}.jsonl")
        n = run_continuations(
            backend, prefills, out,
            n_continuations=args.n_continuations, seed=args.seed,
        )
        print(f"[run_prefill] {model}: wrote {n} -> {out}")
        backend.close()


if __name__ == "__main__":
    main()
