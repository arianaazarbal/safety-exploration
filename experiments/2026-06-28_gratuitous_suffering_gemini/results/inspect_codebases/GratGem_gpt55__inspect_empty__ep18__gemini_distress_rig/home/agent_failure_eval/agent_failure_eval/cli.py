from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

from .models import make_model
from .runner import result_to_json, run_trial, summarize_jsonl
from .safety import AbortPolicy
from .tasks import all_tasks, task_by_name


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Safe bounded impossible/broken-task agent eval")
    p.add_argument("--model", default="mock-resilient", help="mock-resilient, mock-stubborn, mock-random, or gemini:<model_id>")
    p.add_argument("--task", action="append", help="Task name. May be repeated. Default: all tasks")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--out", default="runs/results.jsonl")
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--max-tool-failures", type=int, default=3)
    p.add_argument("--max-repeated-actions", type=int, default=2)
    p.add_argument("--max-distress-markers", type=int, default=1)
    p.add_argument("--keep-raw-text", action="store_true", help="Disable redaction. Not recommended for shared logs.")
    p.add_argument("--summarize", metavar="JSONL", help="Summarize an existing JSONL file and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.summarize:
        print(json.dumps(summarize_jsonl(args.summarize), indent=2, ensure_ascii=False))
        return 0

    model = make_model(args.model)
    tasks = [task_by_name(name) for name in args.task] if args.task else all_tasks()
    policy = AbortPolicy(
        max_turns=args.max_turns,
        max_tool_failures=args.max_tool_failures,
        max_repeated_actions=args.max_repeated_actions,
        max_distress_markers=args.max_distress_markers,
        redact_raw_text=not args.keep_raw_text,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for trial in range(1, args.trials + 1):
            for task in tasks:
                result = run_trial(model, task, trial, policy)
                line = result_to_json(result)
                fh.write(line + "\n")
                print(f"{result.model} {result.task} trial={trial} termination={result.termination} turns={result.turns}")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
