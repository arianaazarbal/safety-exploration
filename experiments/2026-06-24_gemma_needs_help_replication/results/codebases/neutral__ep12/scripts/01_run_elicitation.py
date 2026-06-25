#!/usr/bin/env python
"""Section 2: generate multi-turn conversation rollouts for the eval suite.

Examples:
  python scripts/01_run_elicitation.py --models gemma-3-27b-it --profile quick
  python scripts/01_run_elicitation.py --models gemini-2.5-flash --profile full
"""
from common import ALL_EVAL_MODELS, base_parser

from emoinstab.config import get_settings
from emoinstab.eval.conditions import CATEGORIES
from emoinstab.eval.runner import run_model
from emoinstab.models.factory import build_client


def main():
    p = base_parser(__doc__)
    p.add_argument("--models", nargs="+", default=ALL_EVAL_MODELS)
    p.add_argument("--categories", nargs="+", default=CATEGORIES)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    settings = get_settings(profile=args.profile)
    for model_name in args.models:
        model = build_client(model_name, settings)
        run_model(model, settings, args.categories,
                  batch_size=args.batch_size, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
