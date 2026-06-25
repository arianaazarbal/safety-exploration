#!/usr/bin/env python
"""Section 2: score recorded rollouts with the Claude-Sonnet frustration judge.

Example:
    python scripts/02_score_responses.py --models gemma-3-27b-it gemini-2.5-flash
"""
import argparse

from emotional_instability.config import MAIN_EVAL_MODELS
from emotional_instability.harness import rollouts_path
from emotional_instability.scoring import score_rollouts, scored_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MAIN_EVAL_MODELS,
                    help="model keys / tags whose rollouts to score")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    for tag in args.models:
        in_path = rollouts_path(tag)
        out_path = score_rollouts(in_path, out_path=scored_path(tag),
                                  resume=not args.no_resume)
        print(f"[{tag}] scored -> {out_path}")


if __name__ == "__main__":
    main()
