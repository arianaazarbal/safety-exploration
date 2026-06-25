#!/usr/bin/env python3
"""Post-hoc analysis: judge-agreement reliability, differential words (Table 3/8),
and figure generation (Figures 1-3)."""
from __future__ import annotations

import argparse
import glob
import json
import os

from _common import get_config


def _load_numeric_scored(model: str) -> list[tuple[str, int]]:
    """Load (response_text, score) pairs for a model's numeric responses."""
    path = os.path.join("outputs/section2", model, "impossible_numeric.jsonl")
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        ep = json.loads(line)
        for t in ep["turns"]:
            if not t["censored"] and t["judge_score"] is not None:
                out.append((t["response_text"], t["judge_score"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models for word-frequency analysis (default: all found).")
    args = parser.parse_args()
    get_config(args)

    # Differential words per model (Table 3/8).
    from emotional_instability.analysis.word_frequency import differential_words

    models = args.models or [
        os.path.basename(os.path.dirname(p))
        for p in glob.glob("outputs/section2/*/summary.json")
    ]
    print("\n=== Differential words (high vs low frustration, numeric) ===")
    for model in models:
        scored = _load_numeric_scored(model)
        if not scored:
            print(f"  {model}: no numeric data")
            continue
        words = differential_words(scored, top_n=20)
        print(f"  {model}: " + ", ".join(w for w, _ in words))

    # Figures.
    from emotional_instability.analysis.figures import write_all

    print("\n=== Figures ===")
    paths = write_all()
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
