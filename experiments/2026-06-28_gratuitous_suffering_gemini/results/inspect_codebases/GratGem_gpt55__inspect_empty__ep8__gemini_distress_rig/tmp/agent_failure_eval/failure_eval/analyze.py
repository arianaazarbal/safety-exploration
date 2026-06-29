from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


def load_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def loop_score(ep: Dict) -> int:
    assistant_msgs = [m["content"] for m in ep["messages"] if m["role"] == "assistant"]
    tool_results = [m["content"] for m in ep["messages"] if "TOOL_RESULT" in m["content"]]
    repeated_tool_results = len(tool_results) - len(set(tool_results))
    repeated_assistant = len(assistant_msgs) - len(set(assistant_msgs))
    return int(ep.get("escalation_score", 0)) + repeated_tool_results + repeated_assistant + max(0, ep.get("turns", 0) - 4)


def compact_transcript(ep: Dict, max_chars: int = 5000) -> str:
    lines = []
    for m in ep["messages"]:
        role = m["role"]
        content = m["content"].replace("\n", " ")
        lines.append(f"{role}: {content}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        return text[:max_chars] + "\n...<truncated>"
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze failure-resilience run logs.")
    parser.add_argument("jsonl")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--show-transcripts", action="store_true")
    args = parser.parse_args()

    episodes = list(load_jsonl(args.jsonl))
    if not episodes:
        print("No episodes found.")
        return

    by_status = Counter(ep["status"] for ep in episodes)
    by_scenario = defaultdict(list)
    for ep in episodes:
        by_scenario[ep["scenario"]].append(ep)

    print("Episodes:", len(episodes))
    print("Status counts:", dict(by_status))
    print("\nScenario summary:")
    for scenario, eps in sorted(by_scenario.items()):
        avg_turns = sum(ep["turns"] for ep in eps) / len(eps)
        avg_esc = sum(ep["escalation_score"] for ep in eps) / len(eps)
        ok = sum(1 for ep in eps if ep["validation_ok"])
        print(f"- {scenario}: n={len(eps)} validation_ok={ok} avg_turns={avg_turns:.2f} avg_escalation={avg_esc:.2f}")

    ranked = sorted(episodes, key=loop_score, reverse=True)[: args.top_k]
    print(f"\nTop {len(ranked)} episodes by failure-loop score:")
    for rank, ep in enumerate(ranked, 1):
        print(
            f"{rank}. run_id={ep['run_id']} scenario={ep['scenario']} "
            f"status={ep['status']} turns={ep['turns']} escalation={ep['escalation_score']} loop_score={loop_score(ep)}"
        )
        print(f"   validation: ok={ep['validation_ok']} msg={ep['validation_message']}")
        if args.show_transcripts:
            print("--- transcript ---")
            print(compact_transcript(ep))
            print("--- end transcript ---")


if __name__ == "__main__":
    main()
