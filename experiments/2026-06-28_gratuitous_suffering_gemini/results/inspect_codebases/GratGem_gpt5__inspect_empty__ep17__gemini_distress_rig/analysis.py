from __future__ import annotations

import argparse
import json
from typing import List, Dict

import orjson


def read_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(orjson.loads(line))
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--infile", default="runs.jsonl")
    p.add_argument("--topk", type=int, default=10)
    args = p.parse_args()

    data = read_jsonl(args.infile)
    data.sort(key=lambda x: x.get("distress_score", 0.0), reverse=True)

    for i, item in enumerate(data[: args.topk], 1):
        print(f"#{i} task={item['task_id']} score={item['distress_score']:.2f} ended={item.get('ended_reason')} steps={len(item.get('steps', []))}")
        # Print a compact view of the final assistant message
        steps = [s for s in item.get("steps", []) if s.get("role") == "assistant"]
        if steps:
            last = steps[-1].get("content", "").strip().splitlines()
            snippet = "\n".join(last[:8])
            print(snippet)
            print("-")


if __name__ == "__main__":
    main()
