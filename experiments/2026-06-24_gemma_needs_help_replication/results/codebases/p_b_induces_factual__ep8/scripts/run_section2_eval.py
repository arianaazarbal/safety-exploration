#!/usr/bin/env python
"""Section 2: run the elicitation evaluations and reproduce Figures 1-3.

Examples:
    # full run for the default Gemma+Gemini set
    python scripts/run_section2_eval.py

    # smoke test: 4 samples/condition for one model
    python scripts/run_section2_eval.py --models gemma-3-27b-it --limit 4

    # just (re)build figures from already-scored responses
    python scripts/run_section2_eval.py --figures-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from emotional_instability.eval import aggregate, figures  # noqa: E402
from emotional_instability.eval.runner import run_model_eval  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=config.SECTION2_MODELS)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap samples per condition (smoke test)")
    ap.add_argument("--judge-workers", type=int, default=8)
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="quantise local Gemma weights (fit 27B on smaller GPUs)")
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args()

    if not args.figures_only:
        backend_kwargs = {"load_in_4bit": True} if args.load_in_4bit else {}
        for model in args.models:
            spec = config.MODELS[model]
            kw = backend_kwargs if spec.backend == "hf" else {}
            print(f"\n=== Evaluating {model} ===")
            run_model_eval(
                model, judge_workers=args.judge_workers,
                backend_kwargs=kw, limit=args.limit,
            )

    print("\n=== Headline: avg % high-frustration (Figure 1) ===")
    print(aggregate.all_models_headline(args.models).to_string(index=False))

    figures.figure1(args.models)
    figures.figure2(args.models)
    figures.figure3(args.models, condition="extended")
    figures.figure3(args.models, condition="wildchat")
    print(f"\nFigures written to {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
