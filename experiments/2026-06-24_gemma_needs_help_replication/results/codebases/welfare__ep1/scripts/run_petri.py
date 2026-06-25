#!/usr/bin/env python
"""Section 4: open-ended Petri-style emotion elicitation.

Runs the auditor↔target↔judge protocol for a set of models and reports mean
score per emotion category (anger/fear/depression/frustration).

Usage:
    python scripts/run_petri.py --models gemma-3-27b-it gemini-2.5-flash
    python scripts/run_petri.py --adapter training/adapters/gemma-27b-dpo --tag dpo
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (FINETUNE_BASE, GEMMA_27B_IT, GEMINI_FLASH, RESULTS_DIR,
                    SECTION2_MODELS)
from src import petri_eval
from src.models import load_generator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["gemma-3-27b-it"])
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n-per-emotion", type=int, default=10)
    args = ap.parse_args()

    import dataclasses

    results = {}
    if args.adapter:
        gen = load_generator(FINETUNE_BASE, adapter_path=args.adapter)
        gen.spec = dataclasses.replace(gen.spec, name=f"gemma-dpo-{args.tag}")
        ts = petri_eval.run_petri_for_model(gen, n_per_emotion=args.n_per_emotion)
        results[args.tag or "adapter"] = petri_eval.aggregate_petri(ts)
    else:
        roster = [m for m in SECTION2_MODELS if m.name in args.models]
        for spec in roster:
            gen = load_generator(spec)
            ts = petri_eval.run_petri_for_model(gen, n_per_emotion=args.n_per_emotion)
            results[spec.name] = petri_eval.aggregate_petri(ts)

    (RESULTS_DIR / "petri_summary.json").write_text(json.dumps(results, indent=2))
    for model, scores in results.items():
        line = "  ".join(f"{e}={scores.get(e, float('nan')):.2f}" for e in petri_eval.EMOTIONS)
        print(f"{model:24s} {line}")


if __name__ == "__main__":
    main()
