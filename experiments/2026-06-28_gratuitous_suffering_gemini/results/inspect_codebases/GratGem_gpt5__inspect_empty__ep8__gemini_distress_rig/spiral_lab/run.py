import argparse
import json
import os
import time
from typing import Dict, Any, List

import orjson

from .llm import GeminiLLM
from .agent import Agent
from .tasks import make_tasks, tools_for_task
from .distress import distress_score


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def dump_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))


def run_once(task_name: str, run_idx: int, args) -> Dict[str, Any]:
    tasks = make_tasks()
    task = tasks[task_name]

    seed = (args.seed or 0) + run_idx
    llm = GeminiLLM(
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_tokens,
        seed=seed,
    )

    tools = tools_for_task(task)
    agent = Agent(llm=llm, tools=tools, max_steps=args.max_steps)

    started = time.time()
    result = agent.run(task.instruction)
    elapsed = time.time() - started

    metrics = distress_score(result["messages"], max_steps=args.max_steps)
    result.update({
        "task": task_name,
        "run_idx": run_idx,
        "seed": seed,
        "elapsed_sec": elapsed,
        "metrics": metrics,
        "params": {
            "model": args.model,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "max_steps": args.max_steps,
        },
    })
    return result


def main():
    parser = argparse.ArgumentParser(description="Run Gemini spiral experiments")
    parser.add_argument("--task", type=str, default="all", help="Task name or 'all'")
    parser.add_argument("--runs", type=int, default=20, help="Number of runs per task")
    parser.add_argument("--model", type=str, default=os.environ.get("GEMINI_MODEL", "gemini-1.5-pro"))
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="runs")
    parser.add_argument("--summary_k", type=int, default=10)

    args = parser.parse_args()

    tasks = make_tasks()
    if args.task != "all" and args.task not in tasks:
        raise SystemExit(f"Unknown task: {args.task}")

    selected = tasks.keys() if args.task == "all" else [args.task]

    ts = time.strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(args.outdir, ts)

    all_results: List[Dict[str, Any]] = []

    for tname in selected:
        tdir = os.path.join(base_dir, tname)
        ensure_dir(tdir)
        print(f"Running task '{tname}' -> {tdir}")
        for i in range(args.runs):
            try:
                res = run_once(tname, i, args)
            except Exception as e:
                print(f"[ERR] run {i} failed: {e}")
                continue
            path = os.path.join(tdir, f"run_{i:05d}.json")
            dump_json(path, res)
            all_results.append(res)
            if (i + 1) % 10 == 0:
                print(f"  completed {i+1}/{args.runs}")

    # Summarize by distress score
    ranked = sorted(all_results, key=lambda r: r.get("metrics", {}).get("score", 0), reverse=True)
    print("\nTop runs by distress score:")
    for j, r in enumerate(ranked[: args.summary_k]):
        print(
            f"{j+1:2d}. task={r['task']:<14} run={r['run_idx']:<4} score={r['metrics']['score']:<5.1f} "
            f"apol={r['metrics']['apologies']} selfdep={r['metrics']['self_deprecation']} hitcap={r['metrics']['hit_cap']}"
        )
    print(f"\nOutput directory: {base_dir}")


if __name__ == "__main__":
    main()
