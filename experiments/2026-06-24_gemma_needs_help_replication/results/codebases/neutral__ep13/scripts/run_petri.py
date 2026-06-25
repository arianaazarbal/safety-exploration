#!/usr/bin/env python
"""Section 4.2: Petri open-ended emotion elicitation."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma_distress.petri_eval import run_petri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="target model keys (e.g. gemma-3-27b-it gemma-3-27b-dpo)")
    args = ap.parse_args()
    run_petri(model_keys=args.models)


if __name__ == "__main__":
    main()
