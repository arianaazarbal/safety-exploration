#!/usr/bin/env python
"""Section 2: elicit + judge distress for the Gemma/Gemini scope models.

Examples:
  python scripts/run_section2_eval.py --models gemma-3-27b-it gemini-2.5-flash
  python scripts/run_section2_eval.py --all
  python scripts/run_section2_eval.py --models gemma-3-27b-it --mode redacted
"""
from __future__ import annotations

import json
import os

from _common import base_parser, make_config

from gemma_distress.config import SECTION2_MODELS
from gemma_distress.eval.analysis import (differential_words, headline_metrics,
                                          per_turn_progression)
from gemma_distress.eval.run_eval import evaluate_model
from gemma_distress.utils.io import read_jsonl


def main():
    p = base_parser("Section 2 elicitation eval")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--all", action="store_true", help="Evaluate all scope models.")
    p.add_argument("--mode", default="standard",
                   choices=["standard", "redacted", "neutral_continuation"])
    p.add_argument("--no-judge", action="store_true")
    args = p.parse_args()

    cfg = make_config(args)
    models = SECTION2_MODELS if args.all else (args.models or ["gemma-3-27b-it"])

    for model in models:
        out_dir = evaluate_model(model, cfg, mode=args.mode, judge=not args.no_judge)
        rows = list(read_jsonl(os.path.join(out_dir, f"rollouts_{args.mode}.jsonl")))
        summary = {
            "headline": headline_metrics(rows),
            "per_turn_extended": per_turn_progression(rows, ["extended"]),
            "per_turn_wildchat": per_turn_progression(rows, ["wildchat"]),
            "differential_words": differential_words(rows),
        }
        with open(os.path.join(out_dir, f"summary_{args.mode}.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[{model}] -> {out_dir}")
        print(json.dumps(summary["headline"]["overall"], indent=2))


if __name__ == "__main__":
    main()
