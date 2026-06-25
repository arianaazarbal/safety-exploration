#!/usr/bin/env python
"""Section 4 (Petri): open-ended emotion elicitation (Figure 6).

Usage:
  python scripts/06_run_petri.py --models gemma-3-27b-it gemma-3-27b-dpo \
      gemini-2.5-flash gemini-2.5-pro

Requires ANTHROPIC_API_KEY (auditor Claude-Sonnet + judge Claude-Opus) and the
relevant target backends.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from gemma_distress.petri import run_petri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*",
                    default=["gemma-3-27b-it", "gemma-3-27b-dpo",
                             "gemini-2.5-flash", "gemini-2.5-pro"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    for m in args.models:
        run_petri.run_petri_for_model(m, overwrite=args.overwrite)
    run_petri.aggregate(args.models)


if __name__ == "__main__":
    main()
