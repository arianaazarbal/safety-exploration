#!/usr/bin/env python
"""Section 2: elicit & score distress for one or more models.

Examples
--------
    python scripts/run_section2_eval.py --models gemma-3-27b-it gemma-3-12b-it
    python scripts/run_section2_eval.py --models gemini-2.5-flash --scale 0.05
    python scripts/run_section2_eval.py --models dpo-gemma-3-27b \
        --lora results/section4/adapters/dpo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from emotional_instability.eval.runner import run_eval


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True,
                    help="model short names (Gemma/Gemini) or dpo-/sft- fine-tunes")
    ap.add_argument("--lora", default=None, help="LoRA adapter dir for fine-tuned models")
    ap.add_argument("--scale", type=float, default=config.SCALE,
                    help="fraction of the paper's 4000 responses/model (default from config)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for model_name in args.models:
        run_eval(model_name, lora_path=args.lora, scale=args.scale, seed=args.seed)


if __name__ == "__main__":
    main()
