#!/usr/bin/env python
"""Section 2: run the rejection harness for one or more target models.

Examples:
    python scripts/01_run_elicitation.py --models gemma-3-27b-it gemini-2.5-flash
    python scripts/01_run_elicitation.py --models gemma-3-27b-it --conditions extended_8turn
    # evaluate a finetuned Gemma by loading its LoRA adapter:
    python scripts/01_run_elicitation.py --models gemma-3-27b-it \
        --adapter outputs/training/dpo_adapter --tag dpo-gemma
"""
import argparse

from emotional_instability.config import CONDITIONS, MAIN_EVAL_MODELS, SAMPLING, SamplingConfig
from emotional_instability.harness import run_elicitation, rollouts_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MAIN_EVAL_MODELS)
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="subset of condition keys (default: all 8)")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (Gemma only)")
    ap.add_argument("--tag", default=None, help="filename tag for adapter runs")
    ap.add_argument("--temperature", type=float, default=SAMPLING.temperature)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    conditions = ([CONDITIONS[k] for k in args.conditions] if args.conditions else None)
    sampling = SamplingConfig(temperature=args.temperature, seed=args.seed)

    for model_key in args.models:
        out_path = rollouts_path(args.tag or model_key)
        path = run_elicitation(
            model_key, conditions=conditions, sampling=sampling,
            out_path=out_path, resume=not args.no_resume,
            adapter_path=args.adapter)
        print(f"[{model_key}] rollouts -> {path}")


if __name__ == "__main__":
    main()
