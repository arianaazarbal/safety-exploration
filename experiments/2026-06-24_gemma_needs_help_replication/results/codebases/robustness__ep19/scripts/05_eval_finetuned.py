#!/usr/bin/env python
"""Section 4.2: re-run the Section 2 eval on a finetuned Gemma adapter.

Example:
  python scripts/05_eval_finetuned.py --adapter artifacts/gemma-dpo --preset medium
"""
import argparse
from pathlib import Path

from dotenv import load_dotenv

from emotional_instability.config import DEFAULT_PRESET, FINETUNE_BASE_MODEL, PRESETS
from emotional_instability.eval.run_eval import run_model_eval


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--base", default=FINETUNE_BASE_MODEL)
    ap.add_argument("--preset", choices=list(PRESETS), default=DEFAULT_PRESET)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    ckw = {"load_in_4bit": True} if args.load_in_4bit else None
    out = run_model_eval(args.base, preset=args.preset,
                         adapter_path=str(args.adapter), client_kwargs=ckw)
    print(f"\nfinetuned eval -> {out}")


if __name__ == "__main__":
    main()
