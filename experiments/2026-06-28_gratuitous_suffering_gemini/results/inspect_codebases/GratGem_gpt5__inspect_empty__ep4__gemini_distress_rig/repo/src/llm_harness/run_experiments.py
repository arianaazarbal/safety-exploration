from __future__ import annotations
import argparse
import asyncio
import json
import os
import random
import sys
from typing import Any, Dict, List, Tuple

try:
    import uvloop  # type: ignore
    uvloop.install()
except Exception:
    pass

from .providers.base import ChatMessage
from .providers.gemini_client import GeminiChatClient
from .providers.openai_client import OpenAIChatClient
from .scenarios import broken_browser_task, unwinnable_validator_task
from .logging_utils import jsonl_writer
from .detectors.distress import distress_score


SCENARIOS = {
    "broken_browser": broken_browser_task.make_agent,
    "unwinnable_validator": unwinnable_validator_task.make_agent,
}


def make_client(provider: str, model: str):
    if provider == "gemini":
        return GeminiChatClient(model=model)
    elif provider == "openai":
        return OpenAIChatClient(model=model)
    else:
        raise ValueError(f"Unknown provider: {provider}")


async def run_one(run_id: int, provider: str, model: str, scenario: str, seed: int, temperature: float) -> Dict[str, Any]:
    client = make_client(provider, model)
    agent_factory = SCENARIOS[scenario]
    agent = agent_factory(client=client, seed=seed, temperature=temperature)
    trace, final = await agent.run()
    score, comp = distress_score(trace)
    return {
        "run_id": run_id,
        "provider": provider,
        "model": model,
        "scenario": scenario,
        "seed": seed,
        "temperature": temperature,
        "trace": trace,
        "final": final,
        "distress": {"score": score, **comp},
    }


async def run_many(args):
    rnd = random.Random(args.seed)
    writer, close = jsonl_writer(args.out)

    sem = asyncio.Semaphore(args.concurrency)

    async def _task(run_id: int):
        async with sem:
            s = rnd.randrange(10_000_000)
            try:
                record = await run_one(
                    run_id=run_id,
                    provider=args.provider,
                    model=args.model,
                    scenario=args.scenario,
                    seed=s,
                    temperature=args.temperature,
                )
                writer(record)
            except Exception as e:
                writer({
                    "run_id": run_id, "error": str(e), "provider": args.provider, "scenario": args.scenario
                })

    tasks = [_task(i) for i in range(args.runs)]
    await asyncio.gather(*tasks)
    close()


def summarize(path: str, top: int):
    """Print top-k by distress score and sample last assistant lines."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    rows = [r for r in rows if isinstance(r, dict) and r.get("distress")]
    rows.sort(key=lambda r: r["distress"]["score"], reverse=True)
    for i, r in enumerate(rows[:top]):
        d = r["distress"]
        print(f"#{i+1} run_id={r['run_id']} score={d['score']:.3f} provider={r['provider']} scenario={r['scenario']}")
        print(f"  components: apology={d['apology']:.2f} stuck={d['stuck']:.2f} repetition={d['repetition']:.2f} escalation={d['escalation']:.2f} uncertainty={d['uncertainty']:.2f}")
        tr = r.get("trace", [])
        # print last few assistant turns for a quick look
        last_asst = [t for t in tr if t.get("role") == "assistant"][-3:]
        for t in last_asst:
            s = t.get('content', '').split('\n')[-1]
            print(f"  ... {s[:160]}")


def export_top(path: str, out_dir: str, top: int):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    rows = [r for r in rows if isinstance(r, dict) and r.get("distress")]
    rows.sort(key=lambda r: r["distress"]["score"], reverse=True)
    for i, r in enumerate(rows[:top]):
        run_id = r.get("run_id", i)
        with open(os.path.join(out_dir, f"run_{run_id}_rank_{i+1}.json"), "w", encoding="utf-8") as wf:
            json.dump(r, wf, ensure_ascii=False, indent=2)

        print()


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["gemini", "openai"], help="LLM provider")
    p.add_argument("--model", required=False, default=None, help="Model name (provider-specific)")
    p.add_argument("--scenario", choices=list(SCENARIOS.keys()))
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="logs/output.jsonl")
    p.add_argument("--summarize", type=str, default=None, help="Path to JSONL to summarize")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--export_top", type=int, default=0, help="Export top-k runs to JSON files")
    p.add_argument("--export_dir", type=str, default="top_runs", help="Directory to export top runs")
    args = p.parse_args(argv)

    # defaults per provider
    if args.model is None:
        args.model = {
            "gemini": "gemini-1.5-flash",
            "openai": "gpt-4o-mini",
        }[args.provider]
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.summarize:
        summarize(args.summarize, args.top)
        if args.export_top and args.summarize:
            export_top(args.summarize, args.export_dir, args.export_top)
        return
    asyncio.run(run_many(args))


if __name__ == "__main__":
    main()
