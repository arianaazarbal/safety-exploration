#!/usr/bin/env python
"""Run the Section 2 distress-elicitation protocol over the in-scope models.

Examples:
  # Full protocol (~4000 responses/model) for the 4 stock models:
  python scripts/run_eval.py

  # Cheap smoke test (~1% of responses), just Gemma-3-27B-it:
  python scripts/run_eval.py --scale 0.01 --models Gemma-3-27B-it

  # Evaluate a DPO finetune (local adapter) alongside the stock instruct model:
  python scripts/run_eval.py --models Gemma-3-27B-it --dpo-adapter runs/dpo
"""
from __future__ import annotations

import argparse

from emotional_instability import config
from emotional_instability.config import (
    ALL_MODELS,
    DEFAULT_SAMPLING,
    STOCK_MODELS,
    finetuned_gemma,
)
from emotional_instability.eval.runner import run_eval


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=None,
                    help="Display names (default: the 4 stock Gemma/Gemini models).")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Multiplier on per-category response counts (default 1.0).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judge-model", default=config.JUDGE_MODEL)
    ap.add_argument("--out-dir", default=config.RESULTS_DIR)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--dpo-adapter", default=None,
                    help="Path to a LoRA adapter to evaluate as 'DPO-Gemma'.")
    ap.add_argument("--sft-adapter", default=None,
                    help="Path to a LoRA adapter to evaluate as 'SFT-Gemma'.")
    args = ap.parse_args()

    if args.models:
        specs = [ALL_MODELS[m] for m in args.models]
    else:
        specs = list(STOCK_MODELS)
    if args.dpo_adapter:
        specs.append(finetuned_gemma("DPO-Gemma-3-27B", args.dpo_adapter))
    if args.sft_adapter:
        specs.append(finetuned_gemma("SFT-Gemma-3-27B", args.sft_adapter))

    cfg = DEFAULT_SAMPLING
    cfg.scale = args.scale

    paths = run_eval(
        specs, cfg=cfg, seed=args.seed, judge_model=args.judge_model,
        out_dir=args.out_dir, max_workers=args.max_workers,
    )
    print("\nResults written:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print("\nNext: python scripts/make_figures.py --results-dir", args.out_dir)


if __name__ == "__main__":
    main()
