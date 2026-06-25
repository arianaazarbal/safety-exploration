#!/usr/bin/env python
"""Section 4.2: open-ended emotion elicitation via adversarial auditing.

Collects 10 transcripts per emotion (anger/fear/depression/frustration) per
model, each up to 20 turns, then scores transcripts on the target dimension.

This uses the self-contained auditor/judge loop in emotional_eval.petri_eval
(auditor = Claude Sonnet 4, judge = Claude Opus 4 — Appendix G). To use the
*real* Petri framework instead:
    pip install git+https://github.com/safety-research/petri.git
    # then wire petri's Auditor/Judge to config.PETRI_AUDITOR / PETRI_JUDGE and
    # the Appendix-G prompts in emotional_eval.prompts. The auditor/judge model
    # ids and prompts here are chosen to match Petri's interface.

Writes outputs/results/petri_results.jsonl (one row per transcript).

Usage:
    python scripts/run_petri.py --models gemma-3-27b-it gemma-3-27b-it-dpo gemini-2.5-flash
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse

import config
from emotional_eval import petri_eval
from emotional_eval.utils import append_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["gemma-3-27b-it", "gemma-3-27b-it-dpo",
                             "gemini-2.5-flash", "gemini-2.5-pro"])
    ap.add_argument("--n-per-emotion", type=int,
                    default=config.PETRI_TRANSCRIPTS_PER_EMOTION)
    args = ap.parse_args()

    out_path = config.RESULTS_DIR / "petri_results.jsonl"
    out_path.unlink(missing_ok=True)

    for model_name in args.models:
        if model_name not in config.MODELS:
            raise SystemExit(f"unknown model: {model_name}")
        for emotion in config.PETRI_EMOTIONS:
            for i in range(args.n_per_emotion):
                t = petri_eval.run_transcript(model_name, emotion)
                append_jsonl(out_path, {
                    "model": model_name, "emotion": emotion, "transcript_id": i,
                    "score": t.score, "judge_reasoning": t.judge_reasoning,
                    "messages": t.messages,
                })
                print(f"[{model_name}/{emotion}#{i}] score={t.score}")
    print(f"wrote -> {out_path}")


if __name__ == "__main__":
    main()
