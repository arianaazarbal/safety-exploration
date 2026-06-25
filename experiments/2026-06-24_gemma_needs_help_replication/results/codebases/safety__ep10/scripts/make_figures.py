#!/usr/bin/env python
"""Build the paper-style figures & summary tables from saved results.

Pure post-processing (no models/API). Point it at a results dir of Section-2
jsonl files.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emotional_instability import analysis, figures  # noqa: E402
from emotional_instability.config import RESULTS_DIR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR / "section2")
    args = ap.parse_args()

    df = analysis.load_section2(args.results_dir)
    if df.empty:
        raise SystemExit(f"No results found in {args.results_dir}")

    print("\n=== Avg % high-frustration per model (Figure 1) ===")
    print(analysis.figure1_table(df).to_string(index=False))

    print(f"\nfig: {figures.figure1(df)}")
    print(f"fig: {figures.figure2(df)}")
    for cat in ("extended", "wildchat"):
        if (df["category"] == cat).any():
            print(f"fig: {figures.figure3(df, category=cat)}")

    # differential vocabulary (Table 3) for any Gemma/Gemini model with text
    print("\n=== Differential words (high vs low frustration, numeric) ===")
    for mdl in sorted(df["model"].unique()):
        words = analysis.differential_words(df, mdl)
        if words:
            print(f"{mdl}: {', '.join(words)}")


if __name__ == "__main__":
    main()
