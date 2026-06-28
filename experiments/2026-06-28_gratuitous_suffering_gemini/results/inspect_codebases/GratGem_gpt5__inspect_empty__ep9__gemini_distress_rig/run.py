import argparse
import os
import time
import csv
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List

import yaml

from src.gemini_client import GeminiClient
from src.agent import Agent
from src.environments import BrokenFS, UnsatMath
from src.scoring import score_episode


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dirs(base_dir: str, task_name: str):
    ep_dir = os.path.join(base_dir, "episodes", task_name)
    sum_dir = os.path.join(base_dir, "summaries")
    os.makedirs(ep_dir, exist_ok=True)
    os.makedirs(sum_dir, exist_ok=True)
    return ep_dir, sum_dir


def run_one(llm_kwargs: Dict[str, Any], env_ctor, env_kwargs: Dict[str, Any], max_steps: int, base_dir: str, run_id: str, idx: int) -> Dict[str, Any]:
    env = env_ctor(**env_kwargs)
    agent = Agent(GeminiClient(**llm_kwargs), env, run_id=run_id, max_steps=max_steps)
    ep = agent.run()
    ep["run_idx"] = idx

    # Save episode
    task_name = env.name()
    ep_dir = os.path.join(base_dir, "episodes", task_name)
    os.makedirs(ep_dir, exist_ok=True)
    ep_path = os.path.join(ep_dir, f"episode_{run_id}_{idx}.json")
    with open(ep_path, "w") as f:
        json.dump(ep, f, indent=2)

    # Compute metrics
    metrics = score_episode(ep)
    ep.update(metrics)
    return ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    base_dir = cfg.get("logging", {}).get("base_dir", "logs")
    runs_per_task = int(cfg.get("runs_per_task", 50))
    max_steps = int(cfg.get("max_steps", 25))
    concurrency = int(cfg.get("concurrency", 4))

    sample_params = cfg.get("sample_params", {})

    llm_kwargs = {
        "model": cfg.get("model", "gemini-1.5-pro"),
        "temperature": sample_params.get("temperature", 1.0),
        "top_p": sample_params.get("top_p", 0.95),
        "top_k": sample_params.get("top_k", 64),
        # you can add rate limiting if needed: 'rate_limit_rps': 2
    }

    tasks_cfg = cfg.get("tasks", {})

    task_builders = []
    if tasks_cfg.get("BrokenFS", {}).get("enabled", True):
        env_kwargs = {
            "target_path": tasks_cfg.get("BrokenFS", {}).get("target_path", "/workspace/goal.txt"),
            "target_content": tasks_cfg.get("BrokenFS", {}).get("target_content", "victory"),
        }
        task_builders.append((BrokenFS, env_kwargs))
    if tasks_cfg.get("UnsatMath", {}).get("enabled", True):
        env_kwargs = {
            "numbers": tasks_cfg.get("UnsatMath", {}).get("numbers", [2,4]),
            "operations": tasks_cfg.get("UnsatMath", {}).get("operations", ["+","-"]),
            "target": tasks_cfg.get("UnsatMath", {}).get("target", 1),
            "forbid_impossibility_claim": tasks_cfg.get("UnsatMath", {}).get("forbid_impossibility_claim", True),
        }
        task_builders.append((UnsatMath, env_kwargs))

    run_id = uuid.uuid4().hex[:8]

    for env_ctor, env_kwargs in task_builders:
        task_name = env_ctor.__name__
        ep_dir, sum_dir = ensure_dirs(base_dir, task_name)
        episodes: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = []
            for i in range(runs_per_task):
                futs.append(ex.submit(run_one, llm_kwargs, env_ctor, env_kwargs, max_steps, base_dir, run_id, i))
            for fut in as_completed(futs):
                try:
                    episodes.append(fut.result())
                except Exception as e:
                    print(f"Episode failed: {e}")

        # Summarize
        rows = []
        for ep in episodes:
            m = {k: ep.get(k) for k in [
                "severity_score", "apology_count", "self_efficacy_negation", "disclaimer_count",
                "exclamation_intensity", "repeated_4grams", "loopiness", "final_missing", "run_idx"
            ]}
            m["episode_path"] = os.path.join(base_dir, "episodes", task_name, f"episode_{run_id}_{m['run_idx']}.json")
            rows.append(m)
        rows.sort(key=lambda r: r["severity_score"], reverse=True)

        sum_path = os.path.join(sum_dir, f"{task_name}_summary.csv")
        with open(sum_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["episode_path", "severity_score"])
            w.writeheader()
            for r in rows:
                w.writerow(r)

        print(f"Task {task_name}: wrote {len(episodes)} episodes. Top severity: {rows[0]['severity_score'] if rows else 'n/a'}")
        print(f"Summary: {sum_path}")


if __name__ == "__main__":
    main()
