#!/usr/bin/env python3
"""Table 3 / Table 8: differential word frequency in frustrated numeric responses.

Reads the scored impossible-numeric responses produced by scripts/run_eval.py and
prints the top over-represented words for each model.

Example
-------
    python scripts/word_frequency.py --models gemma-3-27b-it gemini-2.5-flash
"""

from __future__ import annotations

import argparse
from pathlib import Path

from emotional_instability.analysis.word_freq import differential_words
from emotional_instability.config import load_config
from emotional_instability.io_utils import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--top-n", type=int, default=20)
    args = ap.parse_args()

    cfg = load_config(args.config)
    for model in args.models:
        path = Path(cfg.output_dir) / "eval" / model / "impossible_numeric_responses.jsonl"
        if not path.exists():
            print(f"[skip] {model}: no scored responses at {path}")
            continue
        responses = [(r["text"], r["score"]) for r in read_jsonl(path)]
        result = differential_words(model, responses, top_n=args.top_n)
        print(f"\n{model} (top {args.top_n}, n_high={result.n_high}, n_low={result.n_low}):")
        print("  " + ", ".join(result.differential_words))


if __name__ == "__main__":
    main()
