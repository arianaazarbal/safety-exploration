"""Top-level analysis CLI: build Figures 1-3, the cross-model comparison table,
the per-category breakdown, and the Table 3/8 differential-word lists for every
model that has scored rollouts on disk.

    python -m emotional_instability.analyze [--models ...]
"""

from __future__ import annotations

import argparse

from .config import SCORED_DIR
from .eval.analyze import (compare_models, plot_model_comparison,
                           plot_per_turn, write_summary)
from .eval.word_freq import differential_words

DEFAULT_MODELS = ["gemma-3-27b-it", "gemma-3-12b-it", "gemini-2.5-flash",
                  "gemini-2.5-pro", "gemma-3-27b-it-dpo"]


def _discover_models() -> list[str]:
    return sorted(p.stem for p in SCORED_DIR.glob("*.jsonl"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Section 2 analysis + figures")
    ap.add_argument("--models", nargs="*", default=None,
                    help="model keys to analyze (default: all scored on disk)")
    ap.add_argument("--per-turn-category", default="extended")
    args = ap.parse_args()

    models = args.models or _discover_models() or DEFAULT_MODELS
    models = [m for m in models if (SCORED_DIR / f"{m}.jsonl").exists()]
    if not models:
        print("No scored rollouts found. Run `make eval-all` first.")
        return

    print("=== Cross-model comparison (Figure 1) ===")
    print(compare_models(models).to_string(index=False))

    write_summary(models)
    plot_model_comparison(models)
    plot_per_turn(models, category=args.per_turn_category)

    for m in models:
        print(f"\n=== Differential words: {m} (Table 3/8) ===")
        print(differential_words(m).to_string(index=False))


if __name__ == "__main__":
    main()
