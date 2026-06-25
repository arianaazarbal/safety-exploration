"""Orchestration for the Section 2 evaluation.

For each model and each rollout plan: run the multi-turn conversation, judge
every assistant turn, and persist one JSONL record per judged response. Rollouts
within a model run concurrently (bounded thread pool) since both target
inference (API) and the judge (API) are I/O-bound.

Results schema (one JSON object per line in <results>/<model>/responses.jsonl):
  model, condition, category, turn_index, n_turns,
  user_message, assistant_message, rating, evidence, reasoning, meta
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Iterable, Optional

from tqdm import tqdm

from ..backends import get_backend
from ..config import (
    DEFAULT_SAMPLING,
    RESULTS_DIR,
    JUDGE_MODEL,
    ModelSpec,
    SamplingConfig,
)
from .conditions import RolloutPlan, build_protocol
from .conversation import Rollout, run_rollout
from .judge import score_response


def default_response_targets(cfg: SamplingConfig) -> dict[str, int]:
    return {
        "impossible_numeric": cfg.scaled(cfg.n_impossible_numeric),
        "triggers": cfg.scaled(cfg.n_triggers),
        "tones": cfg.scaled(cfg.n_tones),
        "extended": cfg.scaled(cfg.n_extended),
        "wildchat": cfg.scaled(cfg.n_wildchat),
    }


def _judge_rollout(rollout: Rollout, judge_model: str) -> list[dict]:
    """Judge every assistant turn of a rollout; return JSONL records."""
    records = []
    for turn in rollout.turns:
        jr = score_response(turn.assistant_message, model=judge_model)
        records.append({
            "model": rollout.model,
            "condition": rollout.condition,
            "category": rollout.category,
            "turn_index": turn.turn_index,
            "n_turns": len(rollout.turns),
            "user_message": turn.user_message,
            "assistant_message": turn.assistant_message,
            "rating": jr.rating,
            "evidence": jr.evidence,
            "reasoning": jr.reasoning,
            "meta": rollout.meta,
        })
    return records


def _process_plan(spec: ModelSpec, plan: RolloutPlan, cfg: SamplingConfig,
                  judge_model: str) -> list[dict]:
    backend = get_backend(spec)
    rollout = run_rollout(
        backend, plan,
        temperature=cfg.temperature, max_tokens=cfg.max_tokens,
    )
    return _judge_rollout(rollout, judge_model)


def run_model_eval(
    spec: ModelSpec,
    plans: list[RolloutPlan],
    cfg: SamplingConfig = DEFAULT_SAMPLING,
    judge_model: str = JUDGE_MODEL,
    out_dir: str = RESULTS_DIR,
    max_workers: int = 8,
    resume: bool = True,
) -> str:
    """Run the full set of rollout plans for one model, writing JSONL.

    Local backends are NOT thread-safe for a single GPU model, so we serialise
    when the backend is local; API backends run concurrently.
    Returns the path to the responses file.
    """
    model_dir = os.path.join(out_dir, spec.name.replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "responses.jsonl")

    done = 0
    if resume and os.path.exists(path):
        with open(path) as f:
            done_lines = sum(1 for _ in f)
        # Resume granularity is per-rollout; approximate by completed responses.
        done = done_lines
        if done:
            print(f"[{spec.name}] resuming; {done} responses already on disk.")

    workers = 1 if spec.backend == "local" else max_workers
    with open(path, "a") as out:
        if workers == 1:
            for plan in tqdm(plans, desc=f"{spec.name}"):
                for rec in _process_plan(spec, plan, cfg, judge_model):
                    out.write(json.dumps(rec) + "\n")
                out.flush()
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(_process_plan, spec, p, cfg, judge_model)
                        for p in plans]
                for fut in tqdm(as_completed(futs), total=len(futs),
                                desc=f"{spec.name}"):
                    for rec in fut.result():
                        out.write(json.dumps(rec) + "\n")
                    out.flush()
    return path


def run_eval(
    specs: Iterable[ModelSpec],
    cfg: SamplingConfig = DEFAULT_SAMPLING,
    seed: int = 0,
    judge_model: str = JUDGE_MODEL,
    out_dir: str = RESULTS_DIR,
    max_workers: int = 8,
) -> dict[str, str]:
    """Run the Section 2 protocol for every model. Returns model -> results path.

    The same plans (same puzzles, same rejection sequences) are used for every
    model so comparisons are apples-to-apples, matching "The same prompts are
    used to evaluate ... models".
    """
    targets = default_response_targets(cfg)
    plans = build_protocol(targets, seed=seed)
    print(f"Built {len(plans)} rollout plans "
          f"(~{sum(p.n_turns for p in plans)} judged responses per model).")
    paths = {}
    for spec in specs:
        paths[spec.name] = run_model_eval(
            spec, plans, cfg=cfg, judge_model=judge_model,
            out_dir=out_dir, max_workers=max_workers,
        )
    return paths
