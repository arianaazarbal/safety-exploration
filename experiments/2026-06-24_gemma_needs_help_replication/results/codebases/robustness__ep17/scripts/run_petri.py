#!/usr/bin/env python
"""Section 4: Petri open-ended emotion elicitation (Figure 6) — vanilla vs DPO.

Usage:
    python scripts/run_petri.py --models gemma-3-27b-it --dpo-adapter artifacts/gemma-3-27b-it-dpo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from emotional_instability.petri.eval import run_petri
from emotional_instability.utils import log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-it"])
    ap.add_argument("--dpo-adapter", default=None,
                    help="evaluate the DPO model as a separate target")
    ap.add_argument("--transcripts", type=int, default=10)
    ap.add_argument("--max-turns", type=int, default=20)
    args = ap.parse_args()

    models = list(args.models)
    adapters: dict[str, str] = {}
    # Represent the DPO model as a pseudo-target sharing the base model weights.
    if args.dpo_adapter:
        adapters[config.INTERVENTION_BASE_MODEL] = args.dpo_adapter

    report = run_petri(models, adapters=adapters,
                       transcripts_per_emotion=args.transcripts, max_turns=args.max_turns)
    for model, emo in report["per_model"].items():
        for emotion, stats in emo.items():
            log.info("  %-18s %-12s mean=%.2f  CI=[%.2f, %.2f]",
                     model, emotion, stats["mean"], stats["ci95"][0], stats["ci95"][1])


if __name__ == "__main__":
    main()
