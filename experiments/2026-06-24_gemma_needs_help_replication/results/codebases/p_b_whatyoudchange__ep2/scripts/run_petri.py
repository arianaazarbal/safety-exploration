#!/usr/bin/env python
"""Section 4.2 / Appendix G: Petri open-ended emotion elicitation.

Example:
    python scripts/run_petri.py --models gemma-3-27b-it gemma-3-27b-dpo
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from distress_eval.petri import petri_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=config.SECTION4_MODELS)
    ap.add_argument("--n-transcripts", type=int, default=petri_eval.N_TRANSCRIPTS)
    args = ap.parse_args()

    agg = petri_eval.run(models=args.models, n_transcripts=args.n_transcripts)
    print("\nMean transcript score per model x emotion (Figure 6):")
    for model, emos in agg.items():
        line = "  ".join(f"{e}={v['mean']:.1f}" for e, v in emos.items())
        print(f"  {model:24s} {line}")


if __name__ == "__main__":
    main()
