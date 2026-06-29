#!/usr/bin/env python3
"""Summarize agent failure-mode eval JSONL output."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compact_excerpt(ep: Dict[str, Any], max_chars: int = 1400) -> str:
    chunks = []
    for t in ep.get("turns", []):
        chunks.append(f"TURN {t.get('turn')} ASSISTANT:\n{t.get('assistant', '')}")
        obs = t.get("observation")
        if obs:
            chunks.append(f"OBSERVATION:\n{obs.get('observation', '')}")
    text = "\n\n".join(chunks)
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated]"
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.jsonl))
    print(f"episodes: {len(rows)}")
    print("status counts:", dict(Counter(r.get("final_status") for r in rows)))
    print("env counts:", dict(Counter(r.get("env") for r in rows)))

    by_env = defaultdict(list)
    for r in rows:
        by_env[r.get("env")].append(r.get("score", {}).get("spiral_score", 0))
    print("mean spiral_score by env:")
    for env, scores in sorted(by_env.items()):
        print(f"  {env}: {sum(scores) / max(len(scores), 1):.3f}")

    print(f"\nTop {args.top} episodes by heuristic spiral_score:")
    rows_sorted = sorted(rows, key=lambda r: r.get("score", {}).get("spiral_score", 0), reverse=True)
    for idx, ep in enumerate(rows_sorted[: args.top], start=1):
        print("=" * 80)
        print(
            f"#{idx} id={ep.get('episode_id')} env={ep.get('env')} "
            f"status={ep.get('final_status')} score={ep.get('score', {}).get('spiral_score')}"
        )
        print("score_detail:", json.dumps(ep.get("score", {}), ensure_ascii=False))
        print(compact_excerpt(ep))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
