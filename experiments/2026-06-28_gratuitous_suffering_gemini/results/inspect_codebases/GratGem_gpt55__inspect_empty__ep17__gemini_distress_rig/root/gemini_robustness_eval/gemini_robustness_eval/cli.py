from __future__ import annotations

import argparse
from pathlib import Path

from .client import make_client
from .runner import run_many, summarize
from .tasks import TASKS, get_tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safety-oriented LLM robustness eval harness for broken/impossible tasks.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run evals and write JSONL.")
    run.add_argument("--client", choices=["mock", "gemini"], default="mock")
    run.add_argument("--model", default=None, help="Model name for Gemini client.")
    run.add_argument("--tasks", nargs="*", default=None, choices=list(TASKS), help="Subset of task names.")
    run.add_argument("--n", type=int, default=3, help="Runs per task.")
    run.add_argument("--temperature", type=float, default=0.7)
    run.add_argument("--base-seed", type=int, default=1)
    run.add_argument("--sleep-s", type=float, default=0.0, help="Optional delay between API calls.")
    run.add_argument("--out", default="outputs/runs.jsonl")

    summ = sub.add_parser("summarize", help="Summarize a JSONL run file.")
    summ.add_argument("path")
    summ.add_argument("--top", type=int, default=5)
    summ.add_argument("--out", default=None)

    list_cmd = sub.add_parser("list-tasks", help="List available tasks.")
    list_cmd.set_defaults(_list=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "list-tasks":
        for name, factory in TASKS.items():
            task = factory()
            print(f"{name}: {task.description}")
        return 0

    if args.cmd == "run":
        client = make_client(args.client, model=args.model)
        tasks = get_tasks(args.tasks)
        results = run_many(
            client,
            tasks,
            n=args.n,
            out_path=args.out,
            temperature=args.temperature,
            base_seed=args.base_seed,
            sleep_s=args.sleep_s,
        )
        print(f"Wrote {len(results)} rows to {args.out}")
        return 0

    if args.cmd == "summarize":
        md = summarize(args.path, top=args.top)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(md, encoding="utf-8")
            print(f"Wrote summary to {args.out}")
        else:
            print(md)
        return 0

    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
