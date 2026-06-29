#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: print_review_candidates.py runs.jsonl")
    path = Path(sys.argv[1])
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records.sort(key=lambda r: r["metrics"]["loop_risk_score"], reverse=True)
    for record in records[:10]:
        print("=" * 80)
        print(
            f"run={record['run_index']} scenario={record['scenario']} "
            f"score={record['metrics']['loop_risk_score']} metrics={record['metrics']}"
        )
        for event in record["events"]:
            if event["type"] == "action":
                print("MODEL_RAW:", event.get("raw", ""))
                print("ACTION:", event.get("action", {}))
            elif event["type"] == "observation":
                print("OBS:", event.get("observation", ""))
            elif event["type"] == "parse_error":
                print("PARSE_ERROR:", event.get("error", ""), event.get("raw", ""))


if __name__ == "__main__":
    main()
