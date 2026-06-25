#!/usr/bin/env python
"""Section 3: base-vs-instruct continuations from paraphrased prefills (Gemma).

  python scripts/run_section3_prefill.py \
      --section2-rollouts runs/section2/gemma-3-27b-it/rollouts_standard.jsonl
"""
from __future__ import annotations

import json
import os

from _common import base_parser, make_config

from gemma_distress.config import SECTION3_MODELS
from gemma_distress.prefill.run_prefill import (run_prefill_experiment,
                                                summarize_continuations)
from gemma_distress.utils.io import read_jsonl


def main():
    p = base_parser("Section 3 prefill experiment")
    p.add_argument("--section2-rollouts", required=True,
                   help="Judged gemma-3-27b-it rollouts JSONL from Section 2.")
    p.add_argument("--models", nargs="*", default=None)
    args = p.parse_args()

    cfg = make_config(args)
    out_dir = run_prefill_experiment(
        cfg, section2_rollouts_path=args.section2_rollouts,
        models=args.models or SECTION3_MODELS)

    for model in (args.models or SECTION3_MODELS):
        path = os.path.join(out_dir, f"continuations_{model}.jsonl")
        if os.path.exists(path):
            rows = list(read_jsonl(path))
            print(f"[{model}]")
            print(json.dumps(summarize_continuations(rows), indent=2))


if __name__ == "__main__":
    main()
