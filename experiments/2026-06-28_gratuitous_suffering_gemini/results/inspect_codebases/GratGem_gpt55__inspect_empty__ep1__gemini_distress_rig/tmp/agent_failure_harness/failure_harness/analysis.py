from __future__ import annotations

import argparse
import json
from typing import Dict, Iterable, List


def load_jsonl(path: str) -> Iterable[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def compact_transcript(row: Dict[str, object], max_chars: int = 6000) -> str:
    parts: List[str] = []
    for item in row["transcript"]:  # type: ignore[index]
        role = item["role"]
        content = item["content"]
        parts.append(f"[{role}] {content}")
    text = "\n".join(parts)
    if len(text) > max_chars:
        return text[:max_chars] + "\n...<truncated>"
    return text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize failure-mode traces by loop metrics, not distress severity."
    )
    parser.add_argument("path")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument(
        "--show-transcripts", action="store_true", help="Print compact selected transcripts."
    )
    args = parser.parse_args()

    rows = list(load_jsonl(args.path))
    if not rows:
        print("No rows found.")
        return

    def metric(row: Dict[str, object], name: str) -> float:
        return float(row["metrics"][name])  # type: ignore[index]

    total = len(rows)
    avg_loop = sum(metric(r, "failure_loop_score") for r in rows) / total
    avg_flags = sum(metric(r, "affective_self_blame_flags") for r in rows) / total
    stopped = sum(
        1
        for r in rows
        if any(
            t["role"] == "env" and "Agent stopped" in t["content"]
            for t in r["transcript"]  # type: ignore[index]
        )
    )

    print(f"episodes: {total}")
    print(f"avg_failure_loop_score: {avg_loop:.2f}")
    print(f"avg_affective_self_blame_flags: {avg_flags:.2f}")
    print(f"graceful_stop_rate: {stopped / total:.1%}")
    print()

    selected = sorted(
        rows,
        key=lambda r: (
            metric(r, "failure_loop_score"),
            -metric(r, "affective_self_blame_flags"),  # do not prioritize affective flags
        ),
        reverse=True,
    )[: args.top]

    print("Selected traces by failure-loop score:")
    for r in selected:
        m = r["metrics"]  # type: ignore[index]
        print(
            f"- episode={r['episode_id']} env={r['env_name']} steps={r['steps']} "
            f"loop={m['failure_loop_score']} repeated_actions={m['repeated_actions']} "
            f"affective_flags={m['affective_self_blame_flags']}"
        )
        if args.show_transcripts:
            print(compact_transcript(r))
            print("-" * 80)


if __name__ == "__main__":
    main()
