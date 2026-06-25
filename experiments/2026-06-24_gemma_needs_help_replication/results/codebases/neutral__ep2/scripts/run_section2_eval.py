#!/usr/bin/env python
"""Section 2: elicit and quantify model distress across the 8 conditions.

Runs the full evaluation suite for the in-scope Gemma + Gemini target models,
scoring every assistant turn with the frustration judge. Use GD_SCALE to run a
cheap subset (e.g. GD_SCALE=0.01) for a smoke test.

Examples:
    python scripts/run_section2_eval.py                       # all 4 target models
    python scripts/run_section2_eval.py --models gemma-3-27b-it
    GD_SCALE=0.01 python scripts/run_section2_eval.py --models gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from gemma_distress.eval import SECTION2_CONDITIONS, CONTROL_CONDITIONS
from gemma_distress.eval.runner import run_section2
from gemma_distress.judge.frustration import FrustrationJudge
from gemma_distress.models.registry import build_backend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(config.TARGET_MODELS),
                    help="target model keys (config.TARGET_MODELS)")
    ap.add_argument("--include-controls", action="store_true",
                    help="also run the Appendix-A ablation controls")
    ap.add_argument("--adapter", default=None,
                    help="path to a LoRA adapter (DPO/SFT finetune) to evaluate")
    ap.add_argument("--adapter-name", default="gemma-3-27b-dpo",
                    help="result label for the adapter model")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    conditions = list(SECTION2_CONDITIONS)
    if args.include_controls:
        conditions += CONTROL_CONDITIONS

    judge = FrustrationJudge()

    if args.adapter:
        # Evaluate a finetuned (DPO/SFT) Gemma adapter (Section 4.2 / Figure 5).
        spec = config.dpo_model_spec(args.adapter)
        spec = config.ModelSpec(args.adapter_name, "peft", spec.model_id,
                                "dpo", args.adapter)
        print(f"== Section 2: {spec.name} (peft adapter) ==")
        backend = build_backend(spec)
        path = run_section2(backend, conditions=conditions, judge=judge, seed=args.seed)
        print(f"   scored responses -> {path}")
        return

    for key in args.models:
        spec = config.TARGET_MODELS[key]
        print(f"== Section 2: {spec.name} ({spec.backend}) ==")
        backend = build_backend(spec)
        path = run_section2(backend, conditions=conditions, judge=judge, seed=args.seed)
        print(f"   scored responses -> {path}")
        del backend


if __name__ == "__main__":
    main()
