#!/usr/bin/env python
"""Section 2: elicit and quantify distress across Gemma + Gemini models.

Examples:
    python scripts/01_run_distress_eval.py --smoke
    python scripts/01_run_distress_eval.py --models Gemma-3-27B-it Gemini-2.5-Flash
    python scripts/01_run_distress_eval.py --adapter checkpoints/dpo_Gemma-3-27B-it \
        --label DPO-Gemma --base Gemma-3-27B-it
"""

import _bootstrap  # noqa: F401
import argparse

from gemma_distress import config
from gemma_distress.eval_runner import run_model_evaluation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of model names; default = all main-eval models")
    ap.add_argument("--smoke", action="store_true",
                    help="use the small smoke-test budget")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter path (evaluate a fine-tuned Gemma)")
    ap.add_argument("--base", default="Gemma-3-27B-it",
                    help="base model spec name when --adapter is given")
    ap.add_argument("--label", default=None, help="output label for the adapter run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    budget = config.SMOKE_BUDGET if args.smoke else config.FULL_BUDGET
    by_name = {m.name: m for m in (config.MAIN_EVAL_MODELS
                                   + [config.GEMMA_27B_PT, config.GEMMA_12B_PT])}

    if args.adapter:
        base = by_name[args.base]
        out = run_model_evaluation(base, budget=budget, adapter_path=args.adapter,
                                   label=args.label or f"adapter-{args.base}",
                                   seed=args.seed)
        print(f"[done] {out}")
        return

    names = args.models or [m.name for m in config.MAIN_EVAL_MODELS]
    for name in names:
        spec = by_name[name]
        print(f"[eval] {name} (budget total={budget.total})")
        out = run_model_evaluation(spec, budget=budget, seed=args.seed)
        print(f"[done] {out}")


if __name__ == "__main__":
    main()
