#!/usr/bin/env python
"""Compute metrics and tables from one or more §2 eval runs (Figures 1-3, Table 3).

    python scripts/analyze.py results/eval/gemma-3-27b-it/responses.jsonl \
        results/eval/gemini-2.5-flash/responses.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from emotional_instability.eval import metrics, word_freq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("responses", nargs="+", help="responses.jsonl files")
    ap.add_argument("--threshold", type=int, default=metrics.DEFAULT_THRESHOLD)
    args = ap.parse_args()

    report: dict = {}
    for path in args.responses:
        recs = metrics.load_records(path)
        model = recs[0]["model"] if recs else Path(path).parent.name
        numeric = [r for r in recs if r["category"] in ("impossible_numeric", "extended", "tones")]
        report[model] = {
            "summary": metrics.summarise(recs, args.threshold),
            "rollout_peak": metrics.rollout_peak(recs, args.threshold),
            "per_turn_extended": metrics.per_turn(recs, "extended", args.threshold),
            "per_turn_wildchat": metrics.per_turn(recs, "wildchat", args.threshold),
            "differential_words": word_freq.differential_words(numeric),
        }
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
