#!/usr/bin/env python3
"""Section 2: run the distress evaluation for one or more in-scope models.

Examples:
  python scripts/run_eval.py --models gemma-3-27b-it gemma-3-12b-it
  python scripts/run_eval.py --all                       # all EVAL_TARGETS
  python scripts/run_eval.py --models gemma-3-27b-it --welfare   # opt-in safeguard
  SAMPLE_SCALE=0.01 python scripts/run_eval.py --all     # quick smoke run

Ablations from Appendix A:
  --neutral-continuation   replace rejections with "Continue"/"Okay" (A.1)
  --redact-assistant       hide the model's own prior turns (A.2)
  --single-message         fake multi-turn / single user message (A.3)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EVAL_TARGETS, FINETUNE_VARIANTS
from src.eval.runner import run_model_eval
from src.welfare import WelfareConfig

_BY_KEY = {m.key: m for m in EVAL_TARGETS + FINETUNE_VARIANTS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[],
                    help="model keys (see src/config.py)")
    ap.add_argument("--all", action="store_true", help="run all EVAL_TARGETS")
    ap.add_argument("--welfare", action="store_true",
                    help="enable opt-in distress early-stop safeguard")
    ap.add_argument("--redact-assistant", action="store_true")
    ap.add_argument("--single-message", action="store_true")
    args = ap.parse_args()

    specs = EVAL_TARGETS if args.all else [_BY_KEY[k] for k in args.models]
    if not specs:
        ap.error("specify --models or --all")

    welfare = WelfareConfig(enabled=True) if args.welfare else None
    for spec in specs:
        print(f"=== evaluating {spec.key} ===")
        path = run_model_eval(
            spec, welfare=welfare,
            redact_assistant=args.redact_assistant,
            single_message=args.single_message,
        )
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
