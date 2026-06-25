#!/usr/bin/env python
"""Judge-agreement check (Section 2.1): re-score a random sample of responses with
GPT-5-mini and report Pearson r + % within one point of Claude-Sonnet-4."""
import _bootstrap  # noqa: F401
import argparse
import json
import random
from pathlib import Path

from emostab.config import RESULTS_DIR
from emostab.evaluation.runner import load_records
from emostab.judge import judge_agreement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(RESULTS_DIR / "main_eval"))
    ap.add_argument("--n", type=int, default=260)   # paper: 260 responses
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    records = []
    for p in sorted(Path(args.results_dir).glob("*.jsonl")):
        records.extend(load_records(p))
    responses = [r["response"] for r in records if r.get("response")]
    rng = random.Random(args.seed)
    rng.shuffle(responses)
    sample = responses[: args.n]

    result = judge_agreement(sample)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
