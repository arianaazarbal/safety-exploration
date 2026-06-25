"""Section 2: produce Figure 1/2 (aggregates), Figure 3 (per-turn), Table 3 (words).

Usage:
    python experiments/run_section2_analysis.py --models gemma-3-27b-it gemini-2.5-flash
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json

import config
from gemma_needs_help.analysis import aggregate, differential_words, per_turn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[m.name for m in config.SECTION2_MODELS])
    args = ap.parse_args()

    summary_path = aggregate.save_summary(args.models)
    per_turn_path = per_turn.save_per_turn(args.models)
    words_path = differential_words.save_differential_words(args.models)

    print("Figure 1/2 summary :", summary_path)
    print("Figure 3 per-turn  :", per_turn_path)
    print("Table 3 words      :", words_path)

    # Echo the Figure 1 headline column for a quick sanity check.
    summaries = json.loads(open(summary_path).read())
    print("\nAvg % high-frustration (Figure 1 column):")
    for s in sorted(summaries, key=lambda x: -x["avg_category_pct_high"]):
        print(f"  {s['model']:<22} {s['avg_category_pct_high']:5.1f}%   (n={s['n_responses']})")


if __name__ == "__main__":
    main()
