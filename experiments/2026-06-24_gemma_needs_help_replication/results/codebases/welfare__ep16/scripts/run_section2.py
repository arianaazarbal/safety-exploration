#!/usr/bin/env python
"""Run the Section 2 emotion-elicitation eval for one or more models.

Examples:
  python scripts/run_section2.py --models gemma-3-27b-it gemma-3-12b-it
  python scripts/run_section2.py --models gemini-2.5-flash gemini-2.5-pro
  python scripts/run_section2.py --models dpo-gemma --adapter checkpoints/dpo-gemma
  GEMMA_DISTRESS_SMOKE=1 python scripts/run_section2.py --models gemma-3-12b-it
"""
import argparse

from gemma_distress import config
from gemma_distress.judge import FrustrationJudge
from gemma_distress.models import build_client
from gemma_distress.section2 import run_section2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=list(config.GEMMA_INSTRUCT) + list(config.GEMINI_MODELS))
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter path (for dpo-gemma / sft-gemma).")
    ap.add_argument("--history-mode", default="chat",
                    choices=["chat", "single_message", "redacted"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    judge = FrustrationJudge()
    for model_key in args.models:
        client = build_client(model_key, adapter_path=args.adapter)
        path = run_section2(model_key, client, judge,
                            history_mode=args.history_mode, seed=args.seed)
        print(f"[section2] {model_key} -> {path}")


if __name__ == "__main__":
    main()
