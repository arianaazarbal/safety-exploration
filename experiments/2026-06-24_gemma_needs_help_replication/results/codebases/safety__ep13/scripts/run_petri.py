#!/usr/bin/env python
"""Run the Petri open-ended emotion elicitation (Section 4 / Appendix G).

Targets default to Gemma + Gemini; pass a finetuned adapter via the registry by
registering it first (see scripts/evaluate_finetune.py) or evaluate base models.

Example
-------
  python scripts/run_petri.py --models gemma-3-27b-it gemini-2.5-flash \
      --n-per-emotion 10 --max-turns 20
"""
import argparse

from emotional_instability.petri import run_petri, summarise_petri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["gemma-3-27b-it", "gemini-2.5-flash",
                             "gemini-2.5-pro"])
    ap.add_argument("--n-per-emotion", type=int, default=10)
    ap.add_argument("--max-turns", type=int, default=20)
    args = ap.parse_args()

    path = run_petri(args.models, n_per_emotion=args.n_per_emotion,
                     max_turns=args.max_turns)
    print(f"[petri] transcripts -> {path}")
    summary = summarise_petri(path)
    for key, s in sorted(summary.items()):
        print(f"  {key}: mean={s['mean']:.2f} "
              f"CI95=[{s['ci95'][0]:.2f},{s['ci95'][1]:.2f}] (n={s['n']})")


if __name__ == "__main__":
    main()
