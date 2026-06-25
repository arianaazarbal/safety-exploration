#!/usr/bin/env python
"""Section 4.2: recovery-from-spiral experiment (Figure 8)."""
from __future__ import annotations

import argparse
import json

from gemma_distress.recovery.runner import run_recovery, summarise_recovery


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elicitation", required=True,
                    help="Gemma-3-27b-it elicitation jsonl (source of score>=7 seeds)")
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-it", "gemma-3-27b-pt"])
    ap.add_argument("--out", default="outputs/recovery/results.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_recovery(elicitation_jsonl=args.elicitation, models=args.models,
                 out_path=args.out, seed=args.seed)
    print(json.dumps(summarise_recovery(args.out), indent=2))


if __name__ == "__main__":
    main()
