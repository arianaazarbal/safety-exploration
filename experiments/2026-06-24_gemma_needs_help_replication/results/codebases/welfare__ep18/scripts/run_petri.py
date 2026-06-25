"""Run the Petri open-ended emotion elicitation eval (Section 4 / Appendix G).

python -m scripts.run_petri --models gemma-3-27b-it gemini-2.5-flash \
    --n-per-emotion 10
# DPO-finetuned Gemma:
python -m scripts.run_petri --models gemma-3-27b-it \
    --adapter-path artifacts/gemma-3-27b-it-dpo
"""
from __future__ import annotations

import argparse
import json

from emotional_instability.analysis import plot_petri
from emotional_instability.config import RESULTS_DIR
from emotional_instability.petri import run_petri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-it", "gemini-2.5-flash"])
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--n-per-emotion", type=int, default=10)
    args = ap.parse_args()

    summaries = []
    for key in args.models:
        s = run_petri(key, adapter_path=args.adapter_path, n_per_emotion=args.n_per_emotion)
        print(json.dumps(s, indent=2))
        summaries.append(s)
    plot_petri(summaries, RESULTS_DIR / "petri" / "fig_petri.png")


if __name__ == "__main__":
    main()
