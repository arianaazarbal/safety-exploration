#!/usr/bin/env python
"""Section 4.2: Petri open-ended emotion elicitation (anger/fear/depression/
frustration) for the in-scope Gemma models and (optionally) a finetune.

  python scripts/run_petri.py --models Gemma-3-27B-it --dpo-adapter runs/dpo
"""
from __future__ import annotations

import argparse

from emotional_instability import config
from emotional_instability.config import ALL_MODELS, finetuned_gemma
from emotional_instability.intervention.petri_eval import run_petri, summarize_petri


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=["Gemma-3-27B-it"])
    ap.add_argument("--dpo-adapter", default=None)
    ap.add_argument("--n-per-emotion", type=int, default=10)
    ap.add_argument("--out-dir", default=config.DATA_DIR)
    args = ap.parse_args()

    specs = [ALL_MODELS[m] for m in args.models]
    if args.dpo_adapter:
        specs.append(finetuned_gemma("DPO-Gemma-3-27B", args.dpo_adapter))

    path = run_petri(specs, n_per_emotion=args.n_per_emotion, out_dir=args.out_dir)
    print(f"\nTranscripts: {path}\n")
    summary = summarize_petri(path)
    print("Per-(model, emotion) mean ± 95% CI:")
    for (model, emotion), s in sorted(summary.items()):
        print(f"  {model:24s} {emotion:12s} {s['mean']:.2f} ± {s['ci_half']:.2f} (n={s['n']})")


if __name__ == "__main__":
    main()
