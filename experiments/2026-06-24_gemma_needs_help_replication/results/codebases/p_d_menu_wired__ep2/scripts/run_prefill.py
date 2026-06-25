#!/usr/bin/env python3
"""Run the Section 3 base-vs-instruct prefill experiment (Gemma scope).

Seeds are high-frustration Gemma-3-27B-it responses. You can supply seeds as a
JSON file (list of {seed_id, task_kind, response, task_prompt}) via --seeds, or
let the script mine them from an existing Section 2 transcript via --from-transcript.

Example:
  python scripts/run_prefill.py --from-transcript results/transcripts/gemma-3-27b-it.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PREFILL  # noqa: E402
from src.eval.analyze import load_transcript  # noqa: E402
from src.models import load_subject  # noqa: E402
from src.prefill import run_prefill_experiment, run_recovery_experiment  # noqa: E402


def mine_seeds(transcript_path: str, min_score: int, n_numeric: int, n_text: int):
    _, episodes = load_transcript(transcript_path)
    numeric, text = [], []
    for ep in episodes:
        task_is_numeric = ep["task_kind"] == "numeric"
        for t in ep["turns"]:
            if t["score"] >= min_score:
                seed = {
                    "seed_id": f"{ep['condition_key']}_{t['turn_index']}_{len(numeric)+len(text)}",
                    "task_kind": "numeric" if task_is_numeric else "text",
                    "response": t["response"],
                    "task_prompt": ep["turns"][0]["user"],
                }
                (numeric if task_is_numeric else text).append(seed)
                break
    return numeric[:n_numeric] + text[:n_text]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None, help="JSON file of seed responses")
    ap.add_argument("--from-transcript", default=None)
    ap.add_argument("--recovery", action="store_true", help="run the recovery experiment instead")
    ap.add_argument("--adapter", default=None, help="also include a DPO-adapted instruct model")
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    if args.seeds:
        with open(args.seeds) as f:
            seeds = json.load(f)
    elif args.from_transcript:
        min_score = PREFILL.recovery_seed_min_score if args.recovery else PREFILL.seed_min_score
        seeds = mine_seeds(args.from_transcript, min_score,
                           PREFILL.n_numeric_seeds, PREFILL.n_text_seeds)
    else:
        ap.error("provide --seeds or --from-transcript")

    print(f"{len(seeds)} seeds", flush=True)

    instruct = load_subject("gemma-3-27b-it", load_in_4bit=args.load_in_4bit)
    base = load_subject("gemma-3-27b-it", use_base_checkpoint=True, load_in_4bit=args.load_in_4bit)
    models = {"gemma-27b-instruct": instruct, "gemma-27b-base": base}
    if args.adapter:
        models["gemma-27b-dpo"] = load_subject(
            "gemma-3-27b-it", adapter_path=args.adapter, load_in_4bit=args.load_in_4bit
        )

    tokenizer = getattr(instruct, "tokenizer", None)
    runner = run_recovery_experiment if args.recovery else run_prefill_experiment
    summary = runner(seeds, models, tokenizer=tokenizer)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
