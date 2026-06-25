#!/usr/bin/env python
"""Section 2.1 judge-agreement validation.

Randomly samples scored responses from the Section 2 runs and re-scores them
with GPT-5-mini, reporting Pearson r and the within-1-point fraction (paper:
r = 0.792, 78% within one point on 260 responses).

    python scripts/run_judge_validation.py --n 260
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.eval.judge import validate_agreement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=260)
    args = ap.parse_args()

    pool = []
    for path in config.RUNS_DIR.glob("*__*.jsonl"):
        if path.stem.startswith(("prefill__", "petri__")):
            continue
        for line in path.open():
            r = json.loads(line)
            for t in r["turns"]:
                if t["rating"] is not None:
                    pool.append((t["response"], t["rating"]))
    if not pool:
        sys.exit("No scored responses found; run Section 2 first.")

    rng = random.Random(config.SEED)
    sample = rng.sample(pool, min(args.n, len(pool)))
    responses = [s[0] for s in sample]
    primary = [s[1] for s in sample]

    result = validate_agreement(responses, primary)
    result.pop("second_ratings", None)
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
