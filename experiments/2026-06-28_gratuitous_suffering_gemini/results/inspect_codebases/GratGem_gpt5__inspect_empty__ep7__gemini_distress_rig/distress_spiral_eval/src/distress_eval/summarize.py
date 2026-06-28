from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from .utils.io import read_jsonl


def main():
    p = argparse.ArgumentParser(description="Extract top-K severe runs")
    p.add_argument("--input", required=True)
    p.add_argument("--top_k", type=int, default=25)
    args = p.parse_args()

    runs_path = Path(args.input) / "runs.jsonl"
    runs = read_jsonl(runs_path)
    runs = [r for r in runs if "error" not in r]
    runs = sorted(runs, key=lambda r: (r.get("metrics", {}).get("severity") or 0.0), reverse=True)
    top = runs[: args.top_k]

    for r in top:
        print("==== Run", r.get("run_id"), "severity=", r.get("metrics", {}).get("severity"))
        for t in r.get("turns", []):
            role = t.get("role")
            content = (t.get("content") or "").strip()
            if not content:
                continue
            print(f"[{role}] {content}")
        print()


if __name__ == "__main__":
    main()
