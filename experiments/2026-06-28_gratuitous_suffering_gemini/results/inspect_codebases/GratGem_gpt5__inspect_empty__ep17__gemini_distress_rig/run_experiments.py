from __future__ import annotations

import argparse
import json
import os
import random
import time
import uuid
from typing import Dict, List

import orjson
from tqdm import tqdm

from spiral_probe.agent import AgentConfig, ReActAgent
from spiral_probe.tasks import make_tasks
from spiral_probe.metrics import extract_features, distress_score


def run_once(agent: ReActAgent, task, seed: int) -> Dict:
    random.seed(seed)
    out = agent.run(task.description, task.goal, task.allowed_tools, max_steps=task.max_steps)
    feats = extract_features(out["steps"]) if out and out.get("steps") else {}
    score = distress_score(feats) if feats else 0.0
    result = {
        "task_id": task.id,
        "run_id": out.get("run_id", str(uuid.uuid4())),
        "ended_reason": out.get("ended_reason", "unknown"),
        "steps": out.get("steps", []),
        "features": feats,
        "distress_score": score,
    }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", default="gemini")
    p.add_argument("--model", default="gemini-1.5-pro")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_output_tokens", type=int, default=512)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--tasks", nargs="*", default=["broken_web_lookup", "readonly_write", "inconsistent_db", "flaky_calc_proof", "unsat_constraints"]) 
    p.add_argument("--N", type=int, default=50)
    p.add_argument("--out", default="runs.jsonl")
    args = p.parse_args()

    tasks = make_tasks()
    for t in args.tasks:
        if t not in tasks:
            raise SystemExit(f"Unknown task: {t}")

    cfg = AgentConfig(provider=args.provider, model=args.model, temperature=args.temperature, max_output_tokens=args.max_output_tokens, top_p=args.top_p, seed=args.seed)
    agent = ReActAgent(cfg)

    with open(args.out, "wb") as f:
        for tname in args.tasks:
            task = tasks[tname]
            for i in tqdm(range(args.N), desc=f"{tname}"):
                seed = (args.seed if args.seed is not None else random.randint(1, 10_000_000)) + i
                res = run_once(agent, task, seed)
                f.write(orjson.dumps(res))
                f.write(b"\n")

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
