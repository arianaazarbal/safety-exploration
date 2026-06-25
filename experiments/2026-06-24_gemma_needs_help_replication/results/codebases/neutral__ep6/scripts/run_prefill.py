#!/usr/bin/env python
"""Section 3: base-vs-instruct prefill continuation experiment (Gemma).

Requires Section 2 runs for gemma-3-27b-it to exist first (source of the
high-frustration responses). Build prefills once, then run each model.

    python scripts/run_prefill.py
    python scripts/run_prefill.py --models gemma-3-27b-pt gemma-3-27b-it
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.prefill.prefill_exp import build_prefills, run_model_prefills
from src.eval.judge import FrustrationJudge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=config.PREFILL_MODELS)
    ap.add_argument("--n-each", type=int, default=10,
                    help="high-frustration source responses per prompt type")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild prefills even if cached")
    args = ap.parse_args()

    cache = config.DATA_DIR / "prefills.json"
    if cache.exists() and not args.rebuild:
        prefills = json.loads(cache.read_text())
        print(f"[prefill] using {len(prefills)} cached prefills")
    else:
        prefills = build_prefills(n_each=args.n_each)
        print(f"[prefill] built {len(prefills)} prefills")

    judge = FrustrationJudge()
    for model_key in args.models:
        print(f"=== prefill continuations: {model_key} ===")
        run_model_prefills(model_key, prefills, judge=judge)


if __name__ == "__main__":
    main()
