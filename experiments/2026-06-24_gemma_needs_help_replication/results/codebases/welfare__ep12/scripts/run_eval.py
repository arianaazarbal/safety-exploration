#!/usr/bin/env python
"""Section 2 -- run the elicitation evaluation for one target model.

Examples:
    # Full paper budget (4000 responses) for Gemma-3-27B-it
    python scripts/run_eval.py --model google/gemma-3-27b-it --budget paper

    # Cheap smoke test for a Gemini target
    python scripts/run_eval.py --model google/gemini-2.5-flash --budget smoke

    # All in-scope targets
    python scripts/run_eval.py --all --budget paper
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotional_instability import config
from emotional_instability.data import load_wildchat_prompts
from emotional_instability.evaluate import EvalRunner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="HF id (Gemma) or OpenRouter id (Gemini)")
    ap.add_argument("--all", action="store_true", help="run all in-scope targets")
    ap.add_argument("--adapter", default=None, help="LoRA adapter path (finetuned Gemma)")
    ap.add_argument("--budget", choices=["paper", "smoke"], default="smoke")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-wildchat-download", action="store_true",
                    help="use offline fallback WildChat prompts")
    args = ap.parse_args()

    budget = config.PAPER_BUDGET if args.budget == "paper" else config.SMOKE_BUDGET
    wildchat = (None if args.no_wildchat_download
                else load_wildchat_prompts(n_prompts=20))

    models = config.SECTION2_TARGETS if args.all else [args.model]
    if not models or models == [None]:
        ap.error("provide --model or --all")

    for model in models:
        out_dir = os.path.join(args.out, model.replace("/", "_"))
        kwargs = {}
        if args.adapter:
            from emotional_instability.models import build_backend
            kwargs["target_backend"] = build_backend(model, adapter_path=args.adapter)
        runner = EvalRunner(target_model=model, wildchat_prompts=wildchat, **kwargs)
        summary = runner.run(budget=budget, out_dir=out_dir)
        print(f"\n=== {model} ===")
        print(json.dumps({"avg_pct_high": summary["avg_pct_high"],
                          "overall_mean": summary["overall_mean"]}, indent=2))


if __name__ == "__main__":
    main()
