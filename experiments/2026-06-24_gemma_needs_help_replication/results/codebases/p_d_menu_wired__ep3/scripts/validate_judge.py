#!/usr/bin/env python3
"""Cross-judge reliability check (Section 2.1): re-score a sample with
GPT-5-mini and report Pearson r and % within one point.

Example:
  python scripts/validate_judge.py runs/elicitation/gemma-3-27b-it.raw.jsonl
"""
import _bootstrap  # noqa: F401
import argparse
import json

from emotional_instability.analysis import load_episodes
from emotional_instability.config import load_config
from emotional_instability.judge.validation import validate_judge


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    episodes = load_episodes(args.episodes)
    responses = [{"response": t["response"], "score": t["score"]}
                 for ep in episodes for t in ep["turns"]]
    result = validate_judge(cfg, responses)
    print(json.dumps({"n": result.n, "pearson_r": result.pearson_r,
                      "within_one_point": result.within_one_point}, indent=2))


if __name__ == "__main__":
    main()
