#!/usr/bin/env python
"""Petri-style open-ended emotion elicitation (Section 4.1 / Appendix G).

  python scripts/run_petri.py --models gemma-3-27b-it gemini-2.5-flash
  python scripts/run_petri.py --models gemma-3-27b-it --adapter outputs/dpo --tag dpo
"""
import _bootstrap  # noqa: F401

import argparse
import json
import os

import pandas as pd

from emo_instability.models import build_client
from emo_instability.petri import run_petri, score_transcript


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-it"])
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n-transcripts", type=int, default=10)
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--out", default="outputs/petri")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    auditor = build_client("petri-auditor")
    judge = build_client("petri-judge")

    rows = []
    for model_key in args.models:
        target = build_client(model_key, adapter_path=args.adapter)
        name = model_key + (f"-{args.tag}" if args.tag else "")
        transcripts = run_petri(
            target, name, auditor=auditor,
            n_transcripts=args.n_transcripts, max_turns=args.max_turns,
        )
        for t in transcripts:
            score_transcript(t, judge)
            rows.append({"model": name, "emotion_target": t.emotion, **t.scores})
        with open(os.path.join(args.out, f"petri_{name}.jsonl"), "w") as f:
            for t in transcripts:
                f.write(json.dumps({"model": name, "emotion": t.emotion,
                                    "scores": t.scores, "messages": t.messages}) + "\n")

    df = pd.DataFrame(rows)
    if not df.empty:
        summary = df.groupby("model")[["anger", "fear", "depression", "frustration"]].mean()
        print(summary.to_string())
        summary.to_csv(os.path.join(args.out, "petri_summary.csv"))


if __name__ == "__main__":
    main()
