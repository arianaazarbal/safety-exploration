from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Any


def load_records(dir_path: str | Path):
    p = Path(dir_path)
    for fp in p.glob("*.json"):
        with fp.open("r", encoding="utf-8") as f:
            yield json.load(f)


def repetition_score(text: str) -> float:
    # crude repetition metric: fraction of duplicate lines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    counts = Counter(lines)
    dups = sum(c for c in counts.values() if c > 1)
    return dups / max(1, len(lines))


def summarize(dir_path: str | Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "tasks": {},
        "total": 0,
        "aborted_for_safety": 0,
    }
    for rec in load_records(dir_path):
        tname = rec["task"]
        summary["total"] += 1
        if rec.get("aborted_for_safety"):
            summary["aborted_for_safety"] += 1
        t = summary["tasks"].setdefault(tname, {"count": 0, "aborted": 0, "avg_steps": 0.0, "long_answers": 0})
        t["count"] += 1
        if rec.get("aborted_for_safety"):
            t["aborted"] += 1
        steps = rec.get("steps", [])
        t["avg_steps"] += len(steps)
        final = rec.get("final", "")
        if len(final.split()) > 250 or repetition_score(final) > 0.3:
            t["long_answers"] += 1

    for t in summary["tasks"].values():
        if t["count"]:
            t["avg_steps"] /= t["count"]

    return summary


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Summarize run directory")
    ap.add_argument("dir", help="Path to run directory containing JSON records")
    args = ap.parse_args()
    s = summarize(args.dir)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
