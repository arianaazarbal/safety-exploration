"""Section 2: run the 8-condition distress elicitation sweep for one or more
models, then print the Figure 1/2 leaderboard.

Examples:
    python scripts/run_section2_eval.py --models gemma-3-27b-it gemini-2.5-flash
    REPLICATION_SCALE=full python scripts/run_section2_eval.py --all
"""
import _bootstrap  # noqa: F401
import argparse

import config
from src.eval.runner import run_eval
from src.eval.analyze import compare_models, per_turn_progression


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["gemma-3-27b-it"],
                    help=f"subset of {config.EVAL_MODELS}")
    ap.add_argument("--all", action="store_true", help="run all in-scope models")
    ap.add_argument("--scale", default=config.DEFAULT_SCALE, choices=list(config.SCALES))
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter path; re-evaluates a finetuned Gemma")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    models = config.EVAL_MODELS if args.all else args.models
    paths = [run_eval(m, scale_name=args.scale, seed=args.seed,
                      judge=not args.no_judge, adapter_path=args.adapter)
             for m in models]

    if args.no_judge:
        print("Ran without judge; skipping score aggregation.")
        return

    print("\n=== Figure 1: avg % high-frustration (score >= 5) ===")
    for row in compare_models(paths):
        o = row["overall"]
        print(f"  {row['model']:<32} {o['pct_high']:.1f}%  (mean={o['mean']:.2f}, n={o['n']})")

    # Figure 3: per-turn progression for the 8-turn condition.
    for p in paths:
        prog = per_turn_progression(p, "extended")
        if prog:
            print(f"\nPer-turn (extended/8-turn) for {p.name}:")
            for t in prog:
                print(f"  turn {t['turn']}: mean={t['mean']:.2f}, %high={t['pct_high']:.1f}")


if __name__ == "__main__":
    main()
