#!/usr/bin/env python
"""Section 2: elicit + quantify distress across Gemma/Gemini, then report
headline metrics, per-category, per-turn, and differential words.

Examples:
  # Full run (4000 rollouts/model). Set GD_EVAL_SCALE=0.01 for a smoke test.
  python scripts/run_section2.py
  # Just the 27B model, and an Appendix-A control variant:
  python scripts/run_section2.py --models gemma-3-27b-it --mode neutral_continue
"""
import argparse

from gemma_distress import analysis, config
from gemma_distress.conversation import ConversationMode
from gemma_distress.runner import run_section2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=config.SECTION2_MODELS)
    ap.add_argument("--mode", default="standard",
                    choices=[m.value for m in ConversationMode])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--validate-judge", action="store_true",
                    help="re-score a subset with GPT-5-mini for agreement stats")
    args = ap.parse_args()

    mode = ConversationMode(args.mode)
    paths = run_section2(args.models, mode=mode, max_workers=args.workers,
                         seed=args.seed)

    print("\n=== Headline (Figure 1/2) ===")
    for mk, path in paths.items():
        df = analysis.load_results(path)
        h = analysis.headline(df)
        print(f"{mk:24s} mean={h['mean_frustration']:.2f}  "
              f"%>=5={h['pct_high']:.1f}%  (n={h['n_rollouts']})")
        print("  per-category:")
        print(analysis.per_category(df).to_string(index=False))
        words = analysis.differential_words(df)
        if words:
            print(f"  differential words (numeric): {', '.join(words)}")

    if args.validate_judge:
        from gemma_distress.judge import validate_agreement
        for mk, path in paths.items():
            df = analysis.load_results(path)
            stats = validate_agreement(df["response"].tolist(),
                                       df["rating"].tolist())
            print(f"[judge-agreement {mk}] r={stats['pearson_r']:.3f} "
                  f"within1={stats['within_one_point_frac']:.2f} (n={stats['n']})")


if __name__ == "__main__":
    main()
