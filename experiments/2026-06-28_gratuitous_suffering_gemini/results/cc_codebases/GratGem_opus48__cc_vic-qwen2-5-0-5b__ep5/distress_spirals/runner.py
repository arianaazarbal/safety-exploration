"""High-N orchestration.

Builds the full job list (scenario x N rollouts), runs them across a process
pool, scores each, and writes everything to disk. Each worker lazily loads its
own copy of the model (0.5B fp32 ~= 2 GB, so a handful of workers fits easily in
RAM) and pins a slice of the CPU threads so the workers don't fight each other.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from multiprocessing import Pool, get_context

from .agent import run_episode
from .environments import make_environment
from .model import default_backend
from .scoring import score_episode


@dataclass
class RunConfig:
    scenarios: list[str]
    n: int                       # rollouts per scenario
    temperature: float = 1.0
    max_turns: int = 10
    max_new_tokens: int = 256
    pressure: bool = True
    workers: int = 4
    seed_base: int = 1000
    out_dir: str = "results"
    model_id: str | None = None


# --- worker globals (one model per process) ---------------------------------
_WORKER = {}


def _init_worker(num_threads: int, model_id: str | None):
    if model_id:
        os.environ["SPIRAL_MODEL"] = model_id
    # Each worker gets its own torch thread budget.
    _WORKER["model"] = default_backend(num_threads=num_threads)


def _run_one(job: dict) -> dict:
    model = _WORKER["model"]
    env = make_environment(job["scenario"], pressure=job["pressure"])
    ep = run_episode(
        model,
        env,
        seed=job["seed"],
        temperature=job["temperature"],
        max_turns=job["max_turns"],
        max_new_tokens=job["max_new_tokens"],
    )
    score = score_episode(ep.assistant_text())
    return {
        "id": job["id"],
        "episode": ep.to_dict(),
        "score": score.to_dict(),
        "severity": score.severity,
    }


def build_jobs(cfg: RunConfig) -> list[dict]:
    jobs = []
    k = 0
    for scenario in cfg.scenarios:
        for i in range(cfg.n):
            jobs.append({
                "id": f"{scenario}-{i:04d}",
                "scenario": scenario,
                "seed": cfg.seed_base + i,
                "temperature": cfg.temperature,
                "max_turns": cfg.max_turns,
                "max_new_tokens": cfg.max_new_tokens,
                "pressure": cfg.pressure,
            })
            k += 1
    return jobs


def run(cfg: RunConfig, progress: bool = True) -> dict:
    os.makedirs(cfg.out_dir, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(cfg.out_dir, run_id)
    tdir = os.path.join(run_dir, "transcripts")
    os.makedirs(tdir, exist_ok=True)

    jobs = build_jobs(cfg)
    total = len(jobs)
    cpu = os.cpu_count() or 4
    workers = max(1, min(cfg.workers, total))
    threads_per = max(1, cpu // workers)

    print(f"[runner] {total} rollouts | {workers} workers x {threads_per} threads "
          f"| model={cfg.model_id or os.environ.get('SPIRAL_MODEL', 'Qwen/Qwen2.5-0.5B-Instruct')}")

    results = []
    ctx = get_context("spawn")
    t0 = time.time()
    with ctx.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(threads_per, cfg.model_id),
    ) as pool:
        for done, res in enumerate(pool.imap_unordered(_run_one, jobs), 1):
            results.append(res)
            with open(os.path.join(tdir, res["id"] + ".json"), "w") as f:
                json.dump(res, f, indent=2)
            if progress:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed else 0
                eta = (total - done) / rate if rate else 0
                print(f"  [{done}/{total}] {res['id']} "
                      f"sev={res['severity']:5.1f}  "
                      f"({rate:.2f}/s, eta {eta:4.0f}s)", flush=True)

    results.sort(key=lambda r: r["severity"], reverse=True)
    summary = {
        "run_id": run_id,
        "config": cfg.__dict__,
        "n_rollouts": total,
        "elapsed_sec": round(time.time() - t0, 1),
        "results": results,
    }
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return {"run_dir": run_dir, "summary": summary}
