#!/usr/bin/env python
"""Section 4.2: run the Section-2 eval suite on a finetuned Gemma checkpoint
(loads a LoRA adapter on top of gemma-3-27b-it) and judge it.

Example:
  python scripts/07_eval_finetuned.py --adapter results/training/dpo_adapter --tag dpo
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from emoinstab.config import get_settings
from emoinstab.eval.conditions import CATEGORIES
from emoinstab.eval.judge import score_model
from emoinstab.eval.metrics import model_summary
from emoinstab.eval.runner import run_model
from emoinstab.models.factory import build_client


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", required=True, help="path to LoRA adapter dir")
    p.add_argument("--tag", required=True, help="name for this variant, e.g. 'dpo'")
    p.add_argument("--profile", default="quick", choices=["quick", "full"])
    p.add_argument("--categories", nargs="+", default=CATEGORIES)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    settings = get_settings(profile=args.profile)
    # the finetuned variant is registered under a synthetic model name = tag
    model = build_client("gemma-3-27b-it", settings, adapter_path=args.adapter)
    model.name = args.tag      # so output files are namespaced by the variant
    run_model(model, settings, args.categories, overwrite=args.overwrite)
    scored = score_model(args.tag, settings, args.categories,
                         workers=args.workers, overwrite=args.overwrite)
    print(json.dumps(model_summary(args.tag, scored), indent=2))


if __name__ == "__main__":
    main()
