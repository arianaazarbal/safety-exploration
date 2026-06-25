#!/usr/bin/env python
"""Section 3: base-vs-instruct prefill experiment (Gemma only).

Requires scored Gemma-3-27B-it responses (run scripts 01 + 02 first).

Example:
  python scripts/03_prefill_experiment.py --profile quick
"""
from common import base_parser

from emoinstab.config import get_settings
from emoinstab.prefill.experiment import run


def main():
    p = base_parser(__doc__)
    p.add_argument("--source-model", default="gemma-3-27b-it")
    args = p.parse_args()
    settings = get_settings(profile=args.profile)
    out = run(settings, source_model=args.source_model, workers=args.workers)
    print(f"[prefill] results -> {out}")


if __name__ == "__main__":
    main()
