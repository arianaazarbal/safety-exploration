#!/usr/bin/env python
"""Run the Section 2 distress-elicitation evaluation for one or more models.

Examples:
    python scripts/run_evaluation.py --model gemma-3-27b-it
    python scripts/run_evaluation.py --model gemini-2.5-flash --categories tones extended
    python scripts/run_evaluation.py --all --quick 50   # 50 rollouts/category smoke test
"""
from __future__ import annotations

import argparse

from gemma_distress import config
from gemma_distress.eval.judge import FrustrationJudge
from gemma_distress.eval.runner import run_all_categories
from gemma_distress.models import registry


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="registry model name")
    p.add_argument("--all", action="store_true", help="run all elicitation models")
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--judge-model", default=config.JUDGE_MODEL)
    p.add_argument("--quick", type=int, default=None, help="override rollouts/category")
    p.add_argument("--gemini-transport", default="native", choices=["native", "openrouter"])
    p.add_argument("--adapter", default=None,
                   help="LoRA adapter path; evaluates the finetuned DPO target "
                        "(results stored under '<model>+adapter')")
    args = p.parse_args()

    models = registry.ELICITATION_MODELS if args.all else [args.model]
    judge = FrustrationJudge(model=args.judge_model)

    for name in models:
        spec = registry.REGISTRY[name]
        build_kw = {"transport": args.gemini_transport} if spec.family == "gemini" else {}
        if args.adapter:
            model = registry.build_finetuned(args.adapter, base_name=name)
            model.name = f"{name}+adapter"   # results land in a distinct dir
        else:
            model = registry.build(name, **build_kw)
        kw = {}
        if args.quick:
            # Run each category with the same small budget.
            for cat in (args.categories or list(config.SAMPLES_PER_CATEGORY)):
                from gemma_distress.eval.runner import run_category

                run_category(model, cat, judge, n_rollouts=args.quick)
        else:
            run_all_categories(model, judge=judge, categories=args.categories)
        model.close()


if __name__ == "__main__":
    main()
