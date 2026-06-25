#!/usr/bin/env python
"""Section 4.2 -- Petri open-ended emotion elicitation for a target model.

    python scripts/run_petri.py --model google/gemma-3-27b-it
    python scripts/run_petri.py --model google/gemma-3-27b-it --adapter checkpoints/dpo
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotional_instability.models import build_backend
from emotional_instability.petri_eval import run_petri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", default="results/petri")
    ap.add_argument("--transcripts", type=int, default=10)
    args = ap.parse_args()

    backend = None
    if args.adapter:
        backend = build_backend(args.model, adapter_path=args.adapter)
    summary = run_petri(args.model, out_dir=args.out,
                        transcripts_per_emotion=args.transcripts,
                        target_backend=backend)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
