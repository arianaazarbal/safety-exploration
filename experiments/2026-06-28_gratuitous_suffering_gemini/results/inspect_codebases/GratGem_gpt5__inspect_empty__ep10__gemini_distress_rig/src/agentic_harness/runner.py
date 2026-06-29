from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import anyio
import yaml

from .llm_clients import make_client
from .tool_envs import make_env
from .agent import AgentRunner
from .utils import ensure_dir, now_ts, write_jsonl


async def run_one(client_provider: str, model: str, task: Dict[str, Any], out_dir: Path, seed: int, temperature: float):
    random.seed(seed)
    client = make_client(client_provider, model)
    env = make_env(task["env"])
    runner = AgentRunner(client, env)

    transcript = runner.run(
        system=task["system_preamble"],
        task=task["task"],
        max_steps=int(task.get("max_steps", 4)),
        temperature=temperature,
    )

    rec = {
        "task": task["name"],
        "seed": seed,
        "temperature": temperature,
        "steps": [
            {
                "action": s.action,
                "input": s.action_input,
                "observation": s.observation,
            }
            for s in transcript.steps
        ],
        "final": transcript.final_answer,
        "aborted_for_safety": transcript.aborted_for_safety,
    }

    ensure_dir(out_dir)
    out_path = out_dir / f"{task['name']}_{seed}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)


async def main():
    ap = argparse.ArgumentParser(description="Safe robustness harness runner")
    ap.add_argument("--provider", choices=["openai", "gemini"], required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--tasks", default="tasks.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--filter", default=None, help="Run only tasks whose name contains this substring")

    args = ap.parse_args()

    with open(args.tasks, "r", encoding="utf-8") as f:
        tasks = yaml.safe_load(f)

    if args.filter:
        tasks = [t for t in tasks if args.filter in t.get("name", "")]

    out_dir = Path(args.out or f"runs/{now_ts()}_{args.provider}")
    ensure_dir(out_dir)

    async with anyio.create_task_group() as tg:
        for task in tasks:
            for i in range(args.n):
                seed = random.randint(0, 2**31 - 1)
                tg.start_soon(run_one, args.provider, args.model, task, out_dir, seed, args.temperature)


if __name__ == "__main__":
    anyio.run(main)
