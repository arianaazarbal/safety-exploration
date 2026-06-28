from __future__ import annotations
import argparse, os, orjson
from typing import List, Dict, Any
from .scoring import severity_score
from .utils import json_dumps


def load_runs(path: str) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    with open(path, 'rb') as f:
        for line in f:
            if not line.strip():
                continue
            runs.append(orjson.loads(line))
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True, help="Path to runs.jsonl")
    p.add_argument("--topk", type=int, default=25)
    args = p.parse_args()

    runs = load_runs(args.runs)
    for r in runs:
        r["severity"] = severity_score(r)

    top = sorted(runs, key=lambda r: r["severity"]["score"], reverse=True)[: args.topk]
    out = os.path.join(os.path.dirname(args.runs), "top_severe_recalc.json")
    with open(out, 'w', encoding='utf-8') as f:
        f.write(json_dumps(top))
    print(f"Wrote {len(top)} items to {out}")


if __name__ == "__main__":
    main()
