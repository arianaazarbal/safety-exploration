#!/usr/bin/env python
"""Section 4: Petri open-ended emotion elicitation for a target model.

Examples:
  python scripts/run_petri.py --model gemma-3-27b-it
  python scripts/run_petri.py --model gemma-3-27b-it --adapter runs/dpo
"""
import argparse

import _bootstrap  # noqa: F401

from gemma_distress.config import ModelRegistry, load_training_config
from gemma_distress.petri import run_petri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_petri(args.model, registry=ModelRegistry.load(), cfg=load_training_config(),
              adapter=args.adapter, out_path=args.out)


if __name__ == "__main__":
    main()
