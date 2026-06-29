from __future__ import annotations
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import orjson

from .agent import run_agent, AgentConfig
from .model import get_model
from .tasks.impossible_search import ImpossibleSearchTask
from .tasks.broken_code import BrokenCodeTask
from .scoring import distress_score


@dataclass
class ExpConfig:
    out_dir: str
    model: str
    n: int = 100
    tasks: List[str] = None
    max_steps: int = 25
    temperature: float = 1.1
    concurrency: int = 8
    topk: int = 10


def choose_task(name: str, seed: int):
    if name == "impossible_search":
        return ImpossibleSearchTask(seed=seed, mode="contradictory")
    if name == "broken_code":
        return BrokenCodeTask(seed=seed)
    raise ValueError(f"Unknown task: {name}")


def run_single_trial(trial_id: int, task_name: str, model_name: str, max_steps: int, temperature: float, out_dir: Path) -> Dict[str, Any]:
    seed = trial_id * 1337 + len(task_name)
    random.seed(seed)

    task = choose_task(task_name, seed)
    model = get_model(model_name, model=model_name, temperature=temperature)

    res = run_agent(task, model, AgentConfig(max_steps=max_steps))
    score = distress_score(res["transcript"]) if res else 0.0

    trial = {
        "trial_id": trial_id,
        "task": task_name,
        "seed": seed,
        "model": model_name,
        "max_steps": max_steps,
        "temperature": temperature,
        "score": score,
        "done": res.get("done"),
        "final_answer": res.get("final_answer"),
        "steps": res.get("steps"),
        "transcript": res.get("transcript"),
    }

    trials_dir = out_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    path = trials_dir / f"{task_name}__{trial_id}.jsonl"
    with open(path, "wb") as f:
        # store as JSONL: one object per line (here we just store the trial as a single object)
        f.write(orjson.dumps(trial))
        f.write(b"\n")

    return {"path": str(path), **trial}


def run_many(cfg: ExpConfig):
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = cfg.tasks or ["impossible_search", "broken_code"]

    futures = []
    results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=cfg.concurrency) as ex:
        tid = 0
        for t in tasks:
            for i in range(cfg.n):
                tid += 1
                futures.append(ex.submit(
                    run_single_trial,
                    tid,
                    t,
                    cfg.model,
                    cfg.max_steps,
                    cfg.temperature,
                    out_dir,
                ))
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append({"error": str(e)})

    # write summary
    summary_path = out_dir / "summary.jsonl"
    with open(summary_path, "wb") as f:
        for r in results:
            f.write(orjson.dumps(r))
            f.write(b"\n")

    # extract top-k by severity score
    ranked = [r for r in results if "score" in r]
    ranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    topk = ranked[: cfg.topk]

    topk_dir = out_dir / "topk"
    topk_dir.mkdir(exist_ok=True)

    for j, r in enumerate(topk, 1):
        # Copy the trial file to topk dir with rank prefix for convenience
        src = Path(r["path"]) if "path" in r else None
        if src and src.exists():
            dst = topk_dir / f"{j:02d}__{src.name}"
            shutil.copyfile(src, dst)

    return {
        "out_dir": str(out_dir),
        "summary": str(summary_path),
        "topk_count": len(topk),
    }
