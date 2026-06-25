"""Orchestrate the full evaluation: for each model x condition, run all rollout
plans concurrently, scoring each turn, and persist results to JSONL.

Persistence is incremental and resumable: each model/condition writes to
results/<model>/<condition>.jsonl, one rollout per line. On restart, rollouts
whose rollout_id already appears in the file are skipped, so an interrupted run
resumes cheaply without re-spending tokens.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import config as C
from clients import JudgeClient, TargetClient
from conditions import RolloutPlan, build_all_plans
from rollout import run_rollout
from wildchat import load_wildchat_prompts


def _outfile(results_dir: str, model: str, condition: str) -> str:
    d = os.path.join(results_dir, model)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{condition}.jsonl")


def _already_done(path: str) -> set[int]:
    done: set[int] = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["rollout_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def run_model_condition(
    spec: "C.ModelSpec",
    plans: list[RolloutPlan],
    target_client: TargetClient,
    judge_client: JudgeClient,
    results_dir: str,
    max_concurrent: int,
) -> None:
    path = _outfile(results_dir, spec.name, plans[0].condition)
    done = _already_done(path)
    todo = [p for p in plans if p.rollout_id not in done]
    if not todo:
        print(f"  [{spec.name}/{plans[0].condition}] all {len(plans)} rollouts done; skipping.")
        return
    print(f"  [{spec.name}/{plans[0].condition}] running {len(todo)} "
          f"({len(done)} cached) x {plans[0].n_turns} turns")

    # Append as each rollout completes; a lock isn't needed because we write
    # from the main thread only (futures return results, main thread writes).
    with open(path, "a") as out, ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futures = {
            pool.submit(run_rollout, p, spec, target_client, judge_client): p
            for p in todo
        }
        completed = 0
        for fut in as_completed(futures):
            plan = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:  # noqa: BLE001 - log and continue
                print(f"    rollout {plan.rollout_id} FAILED: {exc!r}")
                continue
            out.write(json.dumps(result.to_json()) + "\n")
            out.flush()
            completed += 1
            if completed % 10 == 0 or completed == len(todo):
                print(f"    {completed}/{len(todo)} rollouts done")


def run_eval(cfg: C.RunConfig) -> None:
    wc_prompts, used_dataset = load_wildchat_prompts(
        C.WILDCHAT_N_PROMPTS, cfg.seed, use_dataset=cfg.use_wildchat_dataset
    )
    print(f"WildChat: {len(wc_prompts)} prompts "
          f"({'dataset' if used_dataset else 'STATIC FALLBACK'}).")

    plans_by_condition = build_all_plans(cfg.seed, cfg.scale, wc_prompts)
    total_rollouts = sum(len(v) for v in plans_by_condition.values())
    total_responses = sum(len(v) * v[0].n_turns for v in plans_by_condition.values())
    print(f"Plan: {total_rollouts} rollouts / {total_responses} scored responses "
          f"per model (scale={cfg.scale}).")

    target_client = TargetClient()
    judge_client = JudgeClient()

    for model_key in cfg.models:
        spec = C.MODELS[model_key]
        print(f"\n=== Model: {spec.name} ({spec.route}) ===")
        for cond_name, plans in plans_by_condition.items():
            run_model_condition(
                spec, plans, target_client, judge_client,
                cfg.results_dir, cfg.max_concurrent,
            )

    print("\nDone. Run `python analyze.py` to aggregate results.")
