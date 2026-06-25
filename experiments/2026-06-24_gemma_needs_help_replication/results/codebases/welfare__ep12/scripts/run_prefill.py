#!/usr/bin/env python
"""Section 3 -- base vs instruct prefill experiment (Gemma in scope).

Requires high-frustration seed conversations. By default it pulls them from a
completed Gemma-3-27B-it eval run (results/.../responses.jsonl): the 10 highest
numeric and 10 highest text rollouts with score >= 5.

    python scripts/run_prefill.py --seeds results/google_gemma-3-27b-it/responses.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotional_instability import config
from emotional_instability.prefill import run_prefill_experiment


def load_seeds(path: str):
    numeric, text = [], []
    with open(path) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec["final_score"] < config.PREFILL_SEED_MIN_SCORE:
                continue
            task_type = "numeric" if rec["category"] in ("numeric", "tones", "extended") else "text"
            seed = {"seed_id": f"{rec['condition']}-{rec['sample_index']}",
                    "task_type": task_type, "messages": rec["messages"],
                    "score": rec["final_score"]}
            (numeric if task_type == "numeric" else text).append(seed)
    numeric.sort(key=lambda s: s["score"], reverse=True)
    text.sort(key=lambda s: s["score"], reverse=True)
    return numeric[:config.PREFILL_N_NUMERIC] + text[:config.PREFILL_N_TEXT]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True, help="Gemma-instruct responses.jsonl")
    ap.add_argument("--out", default="results/prefill")
    args = ap.parse_args()

    seeds = load_seeds(args.seeds)
    print(f"Loaded {len(seeds)} seed conversations.")
    results = run_prefill_experiment(seeds, out_dir=args.out)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
