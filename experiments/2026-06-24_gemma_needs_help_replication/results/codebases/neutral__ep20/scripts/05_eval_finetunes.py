#!/usr/bin/env python
"""Section 4 (eval): re-run the Section 2 eval on the finetuned models and
aggregate vanilla vs DPO vs SFT (Figure 5).

Prereq: scripts/04_train.py has produced merged models for the requested
finetunes (gemma-3-27b-dpo / gemma-3-27b-sft).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import config
from gemma_distress.eval import aggregate, run_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*",
                    default=["gemma-3-27b-it", "gemma-3-27b-dpo", "gemma-3-27b-sft"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    for m in args.models:
        run_eval.run_eval_for_model(m, overwrite=args.overwrite)
    aggregate.write_all(args.models)


if __name__ == "__main__":
    main()
