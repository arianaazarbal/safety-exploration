#!/usr/bin/env python
"""Section 3: base-vs-instruct prefilling study (Gemma).

Builds prefills from Gemma-3-27B-it's high-frustration responses, then measures
continuations from the base and instruct 27B models.

    python scripts/run_prefill.py
    python scripts/run_prefill.py --continuations 50 --seed 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from emotional_instability.prefill.experiment import run_prefill_experiment


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(config.PREFILL_MODELS.keys()))
    ap.add_argument("--continuations", type=int,
                    default=config.PREFILL_CONTINUATIONS_PER_PREFILL)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_prefill_experiment(args.models, n_continuations=args.continuations, seed=args.seed)


if __name__ == "__main__":
    main()
