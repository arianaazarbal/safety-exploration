#!/usr/bin/env python
"""Section 2: elicit & quantify distress across the 5 eval categories.

Examples
--------
# full sweep over the default Gemma+Gemini targets
python scripts/run_section2_eval.py

# quick smoke run on one model
python scripts/run_section2_eval.py --models gemma-3-27b-it --profile quick

Requires: HF weights for Gemma targets (GPU); OPENROUTER_API_KEY for Gemini;
ANTHROPIC_API_KEY for the Claude Sonnet 4 judge.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emotional_instability.config import DEFAULT_EVAL_TARGETS, EvalConfig  # noqa: E402
from emotional_instability.eval_runner import run_section2  # noqa: E402
from emotional_instability import analysis  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_EVAL_TARGETS)
    ap.add_argument("--categories", nargs="+", default=None,
                    help="subset of: numeric triggers tones extended wildchat")
    ap.add_argument("--profile", choices=["full", "quick"], default="full")
    ap.add_argument("--no-store-text", action="store_true",
                    help="don't persist response text (saves disk; disables "
                         "differential-word analysis)")
    args = ap.parse_args()

    cfg = EvalConfig.quick() if args.profile == "quick" else EvalConfig()
    paths = {}
    for m in args.models:
        print(f"\n=== Section 2: {m} ===")
        paths[m] = run_section2(m, cfg=cfg, categories=args.categories,
                                store_text=not args.no_store_text)
        print(f"  wrote {paths[m]}")

    # quick aggregate over whatever we just produced
    df = analysis.load_section2(Path(paths[args.models[0]]).parent)
    print("\n--- Figure-1 table (avg % high-frustration) ---")
    print(analysis.figure1_table(df).to_string(index=False))


if __name__ == "__main__":
    main()
