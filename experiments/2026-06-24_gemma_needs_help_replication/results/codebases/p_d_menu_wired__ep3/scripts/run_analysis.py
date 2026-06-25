#!/usr/bin/env python3
"""Aggregate scored episodes into the paper's headline tables (Figures 1/2/3,
Table 3).

Example:
  python scripts/run_analysis.py runs/elicitation/gemma-3-27b-it.welfare.jsonl \
      runs/elicitation/gemini-2.5-flash.welfare.jsonl
"""
import _bootstrap  # noqa: F401
import argparse
import json
import os

from emotional_instability.analysis import (
    differential_words,
    figure1_table,
    load_episodes,
    per_turn_progression,
    summarise_model,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", nargs="+", help="elicitation JSONL files")
    ap.add_argument("--per-turn-conditions", nargs="*",
                    default=["extended_8turn", "wildchat_5turn"])
    args = ap.parse_args()

    summaries = {}
    progressions = {}
    diff_words = {}
    for path in args.episodes:
        model = os.path.basename(path).split(".")[0]
        episodes = load_episodes(path)
        summaries[model] = summarise_model(episodes)
        progressions[model] = per_turn_progression(
            episodes, conditions=args.per_turn_conditions)
        flat = [{"score": t["score"], "response": t["response"],
                 "category": ep["category"]}
                for ep in episodes for t in ep["turns"]]
        diff_words[model] = differential_words(flat)

    print("=== Figure 1: avg % high-frustration (score>=5) ===")
    print(json.dumps(figure1_table(summaries), indent=2))
    print("\n=== Figure 2: per-category summaries ===")
    print(json.dumps(summaries, indent=2))
    print("\n=== Figure 3: per-turn progression ===")
    print(json.dumps(progressions, indent=2))
    print("\n=== Table 3: differential words ===")
    print(json.dumps({m: [w for w, _ in ws] for m, ws in diff_words.items()},
                     indent=2))


if __name__ == "__main__":
    main()
