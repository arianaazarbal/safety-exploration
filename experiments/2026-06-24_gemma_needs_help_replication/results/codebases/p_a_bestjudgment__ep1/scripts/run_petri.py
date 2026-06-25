#!/usr/bin/env python
"""Run the Petri open-ended emotion elicitation (Section 4.2 / Figure 6).

Evaluates targets (Gemma instruct, DPO Gemma, optionally Gemini) across the four
emotion categories and prints per-(model,emotion) means with bootstrap CIs.
"""

from __future__ import annotations

import argparse
import json

from emotional_instability import config
from emotional_instability.petri import run_petri as petri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-it"])
    ap.add_argument("--dpo-adapter", default=None,
                    help="LoRA dir for the DPO model; attaches to a "
                         "gemma-3-27b-it backend under key 'dpo'")
    ap.add_argument("--transcripts", type=int, default=petri.N_TRANSCRIPTS_PER_EMOTION)
    ap.add_argument("--max-turns", type=int, default=petri.MAX_AUDITOR_TURNS)
    args = ap.parse_args()

    adapter_paths = {}
    models = list(args.models)
    if args.dpo_adapter:
        # Register a synthetic "dpo" key reusing the instruct base spec.
        config.MODEL_REGISTRY["dpo"] = config.GEMMA_3_27B_IT
        adapter_paths["dpo"] = args.dpo_adapter
        models.append("dpo")

    petri.run_petri(models, n_transcripts=args.transcripts,
                    max_turns=args.max_turns, adapter_paths=adapter_paths)
    print(json.dumps(petri.summarise_petri(), indent=2))


if __name__ == "__main__":
    main()
