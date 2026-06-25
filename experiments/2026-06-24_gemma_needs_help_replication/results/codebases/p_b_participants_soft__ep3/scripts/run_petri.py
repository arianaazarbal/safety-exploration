#!/usr/bin/env python
"""Section 4.2: Petri open-ended emotion elicitation.

Example:
    python scripts/run_petri.py --model gemma-3-27b-it
    python scripts/run_petri.py --model gemma-3-27b-it --adapter outputs/checkpoints/gemma27b_dpo_all
"""

from __future__ import annotations

import argparse
import json

from emotional_instability.petri.run_petri import run_petri_evaluation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    results = run_petri_evaluation(
        args.model, adapter_path=args.adapter, load_in_4bit=args.load_in_4bit
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
