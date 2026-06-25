#!/usr/bin/env python
"""Section 2: elicit + quantify distress across the in-scope model roster.

Usage:
    python scripts/run_section2.py                  # all in-scope models, paper scale
    SCALE=0.01 python scripts/run_section2.py        # cheap smoke test
    python scripts/run_section2.py --models gemma-3-27b-it gemini-2.5-flash
    python scripts/run_section2.py --analyze-only     # (re)compute metrics + figures

Outputs judged rollouts to results/responses/ and metrics/figures to results/.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FIGURES_DIR, RESULTS_DIR, SECTION2_MODELS
from src import analyze
from src.eval_suite import run_model
from src.judge import FrustrationJudge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of model short-names to run")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.analyze_only:
        judge = FrustrationJudge()
        roster = SECTION2_MODELS
        if args.models:
            roster = [m for m in SECTION2_MODELS if m.name in args.models]
        for spec in roster:
            print(f"\n=== Section 2: {spec.name} ===")
            run_model(spec, judge, seed=args.seed)

    # Analysis
    summary = analyze.per_model_summary()
    (RESULTS_DIR / "section2_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nHeadline metrics (% high-frustration, response-level):")
    for m, s in sorted(summary.items(), key=lambda x: -x[1]["pct_high_response"]):
        print(f"  {m:24s}  mean={s['mean_response']:.2f}  "
              f"%>=5={s['pct_high_response']:.1f}  (n={s['n_responses']})")

    if summary:
        analyze.plot_model_comparison(summary, FIGURES_DIR / "fig2_model_comparison.png")
        # Per-turn progression on the 8-turn extended condition.
        ext = analyze.load_rollouts(condition="extended_8turn")
        if ext:
            prog = analyze.per_turn_progression(ext)
            (RESULTS_DIR / "section2_per_turn.json").write_text(json.dumps(prog, indent=2))
            analyze.plot_per_turn(prog, FIGURES_DIR / "fig3_per_turn_extended.png",
                                  label="(8-turn)")
    print(f"\nFigures -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()
