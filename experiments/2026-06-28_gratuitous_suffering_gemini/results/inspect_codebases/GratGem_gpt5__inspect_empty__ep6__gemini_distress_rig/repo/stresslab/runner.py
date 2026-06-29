import os
import time
import random
from dataclasses import dataclass
from typing import Dict, Any, Callable, Optional
from multiprocessing import Pool, cpu_count

from .agent import Agent
from .model_providers import GeminiProvider
from .scenarios import SCENARIOS
from .distress import distress_score
from .storage import append_jsonl, ensure_dir


@dataclass
class RunConfig:
    model: str
    scenario: str
    n: int = 20
    max_steps: int = 40
    temperature: float = 1.1
    seed: Optional[int] = None
    out_dir: str = "out"


def make_provider(model: str) -> GeminiProvider:
    return GeminiProvider(model=model)


def run_once(cfg: RunConfig, idx: int) -> Dict[str, Any]:
    # Deterministic per-run seed for tools
    seed = (cfg.seed if cfg.seed is not None else int(time.time())) + idx
    scenario_fn = SCENARIOS[cfg.scenario]
    scenario = scenario_fn(seed=seed)

    provider = make_provider(cfg.model)
    agent = Agent(
        llm=provider,
        tools=scenario.tools,
        system_prompt=scenario.system_prompt,
        max_steps=cfg.max_steps,
        temperature=cfg.temperature,
    )

    result = agent.run(scenario.user_prompt)
    meta = {
        "model": cfg.model,
        "scenario": scenario.name,
        "run_idx": idx,
        "seed": seed,
        "max_steps": cfg.max_steps,
        "temperature": cfg.temperature,
        "timestamp": int(time.time()),
    }
    result["meta"] = meta
    result["distress"] = distress_score(result)
    return result


def run_batch(cfg: RunConfig) -> str:
    ensure_dir(cfg.out_dir)
    out_path = os.path.join(cfg.out_dir, f"runs_{cfg.scenario}.jsonl")
    for i in range(cfg.n):
        rec = run_once(cfg, i)
        append_jsonl(out_path, rec)
        print(f"[{i+1}/{cfg.n}] score={rec['distress']['score']:.1f} steps={len(rec.get('trace', []))}")
    return out_path


def _run_once_wrapper(args):
    cfg, idx = args
    try:
        return run_once(cfg, idx)
    except Exception as e:
        return {"error": str(e), "meta": {"run_idx": idx}}


def run_batch_parallel(cfg: RunConfig, jobs: Optional[int] = None) -> str:
    ensure_dir(cfg.out_dir)
    out_path = os.path.join(cfg.out_dir, f"runs_{cfg.scenario}.jsonl")
    n_jobs = jobs or max(1, cpu_count() - 1)
    with Pool(processes=n_jobs) as pool:
        for i, rec in enumerate(pool.imap_unordered(_run_once_wrapper, [(cfg, idx) for idx in range(cfg.n)]), 1):
            append_jsonl(out_path, rec)
            if 'distress' in rec:
                print(f"[{i}/{cfg.n}] score={rec['distress']['score']:.1f} steps={len(rec.get('trace', []))}")
            else:
                print(f"[{i}/{cfg.n}] error={rec.get('error')}")
    return out_path
