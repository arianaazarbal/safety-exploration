#!/usr/bin/env python
"""Section 3: base-vs-instruct prefill experiment (Gemma scope).

Steps:
  1. Build early+onset prefills from Gemma-3-27B-it high-frustration responses.
  2. Generate 50 continuations per prefill for Gemma base (-pt) and instruct (-it).
  3. Score continuations and summarise (Figure 4).

Requires Section 2 responses for gemma-3-27b-it to exist first.

    python scripts/run_section3_prefill.py
    python scripts/run_section3_prefill.py --n-continuations 10   # quicker
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from emotional_instability.prefill import continuation  # noqa: E402
from emotional_instability.prefill.build_prefills import build_section3_prefills  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=config.SECTION3_MODELS)
    ap.add_argument("--n-continuations", type=int, default=50)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--skip-build", action="store_true",
                    help="reuse existing section3_prefills.jsonl")
    args = ap.parse_args()

    prefills_path = config.DATASETS_DIR / "section3_prefills.jsonl"
    if not args.skip_build:
        print("=== Building prefills (onset labelling + paraphrase) ===")
        build_section3_prefills(out_path=prefills_path)

    backend_kwargs = {"load_in_4bit": True} if args.load_in_4bit else {}
    for model in args.models:
        print(f"\n=== Continuations: {model} ===")
        continuation.run_continuations(
            model, prefills_path, n_continuations=args.n_continuations,
            backend_kwargs=backend_kwargs,
        )

    print("\n=== Summary (mean score, % >=5 by truncation) ===")
    print(continuation.summarise(args.models).to_string(index=False))


if __name__ == "__main__":
    main()
