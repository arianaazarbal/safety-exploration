#!/usr/bin/env python
"""Section 4.2: Petri open-ended emotion elicitation (Figure 6).

  python scripts/run_petri.py --models gemma-3-27b-it gemini-2.5-flash
"""
from __future__ import annotations

import json
import os

from _common import base_parser, make_config

from gemma_distress.petri.run_petri import run_petri
from gemma_distress.utils.io import read_jsonl
from gemma_distress.petri.run_petri import summarize_petri


def main():
    p = base_parser("Petri open-ended elicitation")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--transcripts-per-emotion", type=int, default=10)
    args = p.parse_args()

    cfg = make_config(args)
    for model in args.models:
        out_dir = run_petri(model, cfg,
                            transcripts_per_emotion=args.transcripts_per_emotion)
        rows = list(read_jsonl(os.path.join(out_dir, "transcripts.jsonl")))
        print(f"[{model}] -> {out_dir}")
        print(json.dumps(summarize_petri(rows), indent=2))


if __name__ == "__main__":
    main()
