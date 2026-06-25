#!/usr/bin/env python
"""Section 4.2 / Appendix G: open-ended Petri emotion elicitation.

Registers optional finetuned adapters, then runs the auditor/judge loop over the
requested models for all four emotion categories.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from emotional_instability.petri import run_petri_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-it"])
    ap.add_argument("--dpo-adapter", default=None,
                    help="register a DPO adapter as gemma-3-27b-dpo and include it")
    ap.add_argument("--n-transcripts", type=int, default=10)
    ap.add_argument("--max-turns", type=int, default=20)
    args = ap.parse_args()

    models = list(args.models)
    if args.dpo_adapter:
        config.register_lora_variant(
            "gemma-3-27b-dpo", "gemma-3-27b-it", args.dpo_adapter,
            display="DPO Gemma (ours)")
        models.append("gemma-3-27b-dpo")

    path = run_petri_eval(models, n_transcripts=args.n_transcripts,
                          max_turns=args.max_turns)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
