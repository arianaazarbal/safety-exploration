#!/usr/bin/env python
"""Section 4.2: Petri open-ended emotion elicitation.

Targets Gemma/Gemini models (and optionally the DPO finetune via --adapter).

Examples:
  python scripts/08_run_petri.py --models gemma-3-27b-it gemini-2.5-flash
  python scripts/08_run_petri.py --models dpo --adapter results/training/dpo_adapter
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from emoinstab.config import get_settings
from emoinstab.petri.harness import run, summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=["gemma-3-27b-it", "gemini-2.5-flash"])
    p.add_argument("--adapter", default=None,
                   help="LoRA adapter (used when a model name is 'dpo')")
    p.add_argument("--profile", default="quick", choices=["quick", "full"])
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    settings = get_settings(profile=args.profile)
    for name in args.models:
        # support evaluating the DPO finetune as a target (adapter on instruct)
        if name == "dpo" and args.adapter:
            out = run(name, settings, overwrite=args.overwrite,
                      adapter_path=args.adapter, base_model="gemma-3-27b-it")
        else:
            out = run(name, settings, overwrite=args.overwrite)
        print(f"[petri:{name}] {json.dumps(summarize(out), indent=2)}")


if __name__ == "__main__":
    main()
