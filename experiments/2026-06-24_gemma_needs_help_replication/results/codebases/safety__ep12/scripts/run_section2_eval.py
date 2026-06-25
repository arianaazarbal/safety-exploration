#!/usr/bin/env python
"""Section 2: elicit + quantify distress for a target model.

Examples:
  python scripts/run_section2_eval.py --model gemma-3-27b-it
  python scripts/run_section2_eval.py --model gemini-2.5-flash --scale 0.02
  python scripts/run_section2_eval.py --model gemma-3-27b-it --adapter runs/dpo
"""
import argparse

import _bootstrap  # noqa: F401

from gemma_distress.config import ModelRegistry, load_eval_config
from gemma_distress.eval_runner import run_section2
from gemma_distress.rollout import AblationConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="path to a LoRA adapter to load")
    ap.add_argument("--scale", type=float, default=None, help="override eval.yaml scale")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-judge", action="store_true", help="generate only, skip scoring")
    # Appendix A ablations:
    ap.add_argument("--neutral-continuation", action="store_true")
    ap.add_argument("--redacted-turns", action="store_true")
    ap.add_argument("--fake-multiturn", action="store_true")
    args = ap.parse_args()

    registry = ModelRegistry.load()
    eval_cfg = load_eval_config()
    if args.scale is not None:
        eval_cfg["scale"] = args.scale

    ablation = AblationConfig(
        neutral_continuation=args.neutral_continuation,
        redacted_turns=args.redacted_turns,
        fake_multiturn=args.fake_multiturn,
    )
    run_section2(
        args.model, registry=registry, eval_cfg=eval_cfg, ablation=ablation,
        adapter=args.adapter, out_path=args.out, judge_responses=not args.no_judge,
    )


if __name__ == "__main__":
    main()
