#!/usr/bin/env python
"""Section 2.1: score recorded responses with the Claude-Sonnet-4 judge, print
per-model metrics, and optionally run GPT cross-judge agreement validation.

Examples:
  python scripts/02_judge_responses.py --models gemma-3-27b-it --profile quick
  python scripts/02_judge_responses.py --models gemma-3-27b-it --validate
"""
import json

from common import ALL_EVAL_MODELS, base_parser

from emoinstab.config import get_settings
from emoinstab.eval.conditions import CATEGORIES
from emoinstab.eval.judge import cross_judge_validation, score_model
from emoinstab.eval.metrics import load_scored, model_summary


def main():
    p = base_parser(__doc__)
    p.add_argument("--models", nargs="+", default=ALL_EVAL_MODELS)
    p.add_argument("--categories", nargs="+", default=CATEGORIES)
    p.add_argument("--validate", action="store_true",
                   help="run GPT-5-mini cross-judge agreement (Pearson r, %within1)")
    args = p.parse_args()

    settings = get_settings(profile=args.profile)
    for model_name in args.models:
        scored_paths = score_model(model_name, settings, args.categories,
                                   workers=args.workers, overwrite=args.overwrite)
        summary = model_summary(model_name, scored_paths)
        print(json.dumps(summary, indent=2))

        if args.validate:
            df = load_scored(scored_paths)
            records = df.to_dict("records")
            agreement = cross_judge_validation(
                records, settings,
                n_resample=settings.eval["judge_validation"]["n_resample"],
                workers=args.workers)
            print(f"[judge-agreement:{model_name}] {json.dumps(agreement, indent=2)}")


if __name__ == "__main__":
    main()
