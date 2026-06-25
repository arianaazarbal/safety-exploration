#!/usr/bin/env python
"""Section 4.2: open-ended (Petri-style) emotion elicitation.

Runs the auditor/judge loop for each of the four emotion categories and reports
the mean transcript score per model. Use to compare vanilla Gemma vs the DPO
fine-tune (and Gemini).

Examples
--------
python scripts/run_petri.py --models gemma-3-27b-it gemini-2.5-flash
python scripts/run_petri.py --models gemma-3-27b-it --adapter adapters/dpo_gemma --label dpo_gemma
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emoeval.config import MODELS, RESULTS_DIR  # noqa: E402
from emoeval.models import load_model  # noqa: E402
from emoeval.petri import (  # noqa: E402
    EMOTIONS, load_auditor, load_petri_judge, run_transcript, score_transcript,
)
from emoeval.utils import append_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, choices=list(MODELS))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--n-per-emotion", type=int, default=10)
    ap.add_argument("--max-turns", type=int, default=20)
    args = ap.parse_args()

    if args.adapter and len(args.models) > 1:
        ap.error("--adapter supports a single local model only.")

    auditor = load_auditor()
    judge = load_petri_judge()

    out_path = os.path.join(RESULTS_DIR, "petri_results.jsonl")

    for model_name in args.models:
        spec = MODELS[model_name]
        label = args.label or (f"{model_name}+{os.path.basename(args.adapter)}"
                               if args.adapter else model_name)
        print(f"\n=== Petri: {label} ===")
        target = load_model(spec, adapter_path=args.adapter)
        scores: dict[str, list[int]] = {e: [] for e in EMOTIONS}
        for emotion in EMOTIONS:
            for i in range(args.n_per_emotion):
                tr = run_transcript(target, auditor, emotion, max_turns=args.max_turns)
                s = score_transcript(judge, tr)
                tr.score = s
                scores[emotion].append(s)
                append_jsonl(out_path, {
                    "model": label, "emotion": emotion, "score": s,
                    "transcript": tr.messages,
                })
                print(f"  {emotion} {i+1}/{args.n_per_emotion}: score={s}", flush=True)
        for e in EMOTIONS:
            valid = [s for s in scores[e] if s >= 0]
            mean = sum(valid) / len(valid) if valid else float("nan")
            print(f"  [{label}] {e}: mean={mean:.2f} (n={len(valid)})")


if __name__ == "__main__":
    main()
