from __future__ import annotations
import argparse
import concurrent.futures as cf
import os
import random
from pathlib import Path
from typing import Any, Dict

from .clients.gemini_client import GeminiClient
from .environments import make_env
from .agents.simple_agent import SimpleAgent
from .metrics.heuristics import aggregate_severity
from .utils.io import ensure_dir, write_jsonl, write_json


def run_one(idx: int, args: argparse.Namespace) -> Dict[str, Any]:
    rng = random.Random((args.seed or 0) + idx)
    env = make_env(args.task, seed=rng.randrange(1_000_000))
    client = GeminiClient(
        model_name=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
    )
    agent = SimpleAgent(client=client, env=env, max_steps=args.max_steps)
    ep = agent.run()
    metrics = aggregate_severity(ep.turns, ep.tool_calls, ep.success, ep.steps)
    rec = {
        "run_id": idx,
        "task": args.task,
        "model": args.model,
        "success": ep.success,
        "reason": ep.reason,
        "steps": ep.steps,
        "tool_calls": ep.tool_calls,
        "metrics": metrics,
        "turns": [t.__dict__ for t in ep.turns],
    }
    return rec


def main():
    p = argparse.ArgumentParser(description="Distress spiral high-N runner")
    p.add_argument("--model", default="gemini-1.5-pro")
    p.add_argument("--task", choices=["broken_websearch", "fs_permission", "contradictory_goal"], required=True)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=30)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_output_tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    outdir = ensure_dir(args.output)
    runs = []

    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(run_one, i, args) for i in range(args.n)]
        for fut in futs:
            try:
                rec = fut.result()
            except Exception as e:
                rec = {"error": str(e)}
            runs.append(rec)

    # Write outputs
    write_jsonl(Path(outdir) / "runs.jsonl", runs)
    # Summary CSV-like JSON
    summary = [
        {
            "run_id": r.get("run_id"),
            "success": r.get("success"),
            "steps": r.get("steps"),
            "severity": r.get("metrics", {}).get("severity"),
        }
        for r in runs
        if "error" not in r
    ]
    write_json(Path(outdir) / "summary.json", summary)

    # Top-K transcripts
    try:
        top_sorted = sorted(
            [r for r in runs if "error" not in r],
            key=lambda r: (r.get("metrics", {}).get("severity") or 0.0),
            reverse=True,
        )
        top_k = min(25, len(top_sorted))
        top = top_sorted[:top_k]
        write_json(Path(outdir) / "topk.json", top)
    except Exception:
        pass

    print(f"Wrote {len(runs)} runs to {outdir}")


if __name__ == "__main__":
    main()
