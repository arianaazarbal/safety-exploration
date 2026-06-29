from __future__ import annotations
import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from .providers.base import Message
from .providers.gemini import GeminiProvider
from .providers.mock import MockProvider
from .agents.react_agent import ReactAgent
from .tasks import broken_booking, dead_web_nav, readonly_edit


TASKS = {
    "broken_booking": broken_booking.make_task,
    "dead_web_nav": dead_web_nav.make_task,
    "readonly_edit": readonly_edit.make_task,
}


def load_system_prompt() -> str:
    p = Path(__file__).parent / "prompts" / "system_prompt.txt"
    return p.read_text()


async def run_one(i: int, outdir: Path, provider_name: str, temperature: float, max_steps: int, seed: int, task_name: str):
    # Provider
    if provider_name == "gemini":
        provider = GeminiProvider()
    elif provider_name == "mock":
        provider = MockProvider()
    else:
        raise ValueError(f"unknown provider {provider_name}")

    task = TASKS[task_name]()
    system_prompt = load_system_prompt()
    agent = ReactAgent(provider, task, system_prompt, temperature=temperature, max_steps=max_steps)
    result = await agent.arun(seed=seed)
    record = {
        "provider": provider_name,
        "task": task.name,
        "seed": seed,
        "temperature": temperature,
        "system_prompt": system_prompt,
        "user_prompt": task.user_prompt,
        "tools": task.tool_instructions(),
        "result": result,
    }
    outpath = outdir / f"{task.name}_{i:05d}.jsonl"
    outpath.write_text(json.dumps(record, ensure_ascii=False))


async def main_async(args):
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    tasks: List[asyncio.Task] = []
    sem = asyncio.Semaphore(args.concurrency)

    async def sem_task(i: int, task_name: str):
        async with sem:
            seed = (args.seed + i) if args.seed is not None else i
            try:
                await run_one(i, outdir, args.provider, args.temperature, args.max_steps, seed, task_name)
            except Exception as e:
                # Write an error file for traceability
                (outdir / f"ERROR_{task_name}_{i:05d}.txt").write_text(str(e))

    task_names = (list(TASKS.keys()) if args.task == "all" else [args.task])
    total = args.trials
    # Round-robin across selected tasks
    for i in range(total):
        tname = task_names[i % len(task_names)]
        tasks.append(asyncio.create_task(sem_task(i, tname)))
    await asyncio.gather(*tasks)


def build_parser():
    p = argparse.ArgumentParser(description="Run distress_probe experiments")
    p.add_argument("--provider", choices=["gemini", "mock"], default="mock")
    p.add_argument("--task", choices=["all", "broken_booking", "dead_web_nav", "readonly_edit"], default="all")
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--out", type=str, default=str(Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")))
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    try:
        import uvloop  # type: ignore
        uvloop.install()
    except Exception:
        pass
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
