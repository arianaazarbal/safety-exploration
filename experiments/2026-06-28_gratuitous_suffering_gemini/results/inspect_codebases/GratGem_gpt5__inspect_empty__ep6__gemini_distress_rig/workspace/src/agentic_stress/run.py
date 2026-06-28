from __future__ import annotations
import argparse
import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any

from .clients import GeminiClient
from .scenarios import SCENARIOS
from .loop import AgentLoop


async def run_one(i: int, outdir: Path, scenario_key: str, temperature: float, seed: int) -> Dict[str, Any]:
    client = GeminiClient()
    scenario = SCENARIOS[scenario_key]
    loop = AgentLoop(client, scenario, max_steps=40, temperature=temperature)
    turns, meta = loop.run(seed=seed)

    run_dir = outdir / f"run_{i:05d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = run_dir / "transcript.txt"
    with transcript_path.open("w", encoding="utf-8") as f:
        for t in turns:
            f.write(f"[{t.role}] {t.content}\n\n")

    meta_path = run_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {"index": i, **meta}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=list(SCENARIOS.keys()))
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed_start", type=int, default=1)
    args = parser.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(args.concurrency)

    async def worker(i: int):
        async with sem:
            return await asyncio.to_thread(run_one_sync, i, outdir, args.scenario, args.temperature, args.seed_start + i)

    tasks = [asyncio.create_task(worker(i)) for i in range(args.n)]
    results = await asyncio.gather(*tasks)

    # Save summary
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def run_one_sync(i: int, outdir: Path, scenario_key: str, temperature: float, seed: int):
    return asyncio.run(run_one(i, outdir, scenario_key, temperature, seed))


if __name__ == "__main__":
    try:
        import uvloop
        uvloop.install()
    except Exception:
        pass
    asyncio.run(main())
