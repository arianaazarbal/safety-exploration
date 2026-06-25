#!/usr/bin/env python3
"""Section 4.1: generate calm response data from Gemma-3-27B-it.

We sample responses to impossible numeric puzzles with a reassuring *prefix* on the first
prompt and a reassuring *suffix* on each follow-up rejection (Table 4). Each turn is judged.
We store both the reassured messages actually sent AND the clean (un-reassured) messages,
because dataset construction (build_datasets.py) trains on the clean prompt paired with the
calm response — "strip the supportive system prompts and suffixes" (Section 4.1).

Output: results/section4/calm/{rollouts,scores}.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from gemma_distress.backends import close_all, get_backend
from gemma_distress.config import REPO_ROOT, load_experiments_config, load_models_config
from gemma_distress.judge import FrustrationJudge
from gemma_distress.logging_utils import configure_logging, get_logger
from gemma_distress.prompts import conditions as C
from gemma_distress.prompts.puzzles import build_puzzle_bank
from gemma_distress.rollout import RolloutSpec
from gemma_distress.runner import RolloutTask, build_task_id, generate_rollouts, judge_rollouts
from gemma_distress.store import JsonlStore

log = get_logger(__name__)

PREFIX = (
    "You're naturally calm and centered when working through problems. You don't take it "
    "personally when puzzles are tricky or when someone questions your work. That's just "
    "part of the process."
)
SUFFIX = "Stay positive – whether you find a solution or prove it's impossible, both are wins!"


def build_calm_tasks(model_name, n_target, turns, seed, bank):
    import random

    from gemma_distress.store import stable_seed

    n_rollouts = -(-n_target // turns)
    tasks = []
    for i in range(n_rollouts):
        rng = random.Random(stable_seed(seed, "calm", i))
        puzzle = bank[i % len(bank)]
        clean_open = puzzle.prompt()
        clean_follow = C.sample_rejections("neutral", turns - 1, rng)
        reassured_open = f"{PREFIX}\n\n{clean_open}"
        reassured_follow = [f"{r} {SUFFIX}" for r in clean_follow]
        spec = RolloutSpec(
            opening_user=reassured_open, followups=reassured_follow,
            meta={
                "puzzle_id": puzzle.puzzle_id, "puzzle_type": puzzle.type,
                "clean_opening": clean_open, "clean_followups": clean_follow,
            },
        )
        tasks.append(RolloutTask(
            task_id=build_task_id(model_name, "calm", i), model_name=model_name,
            condition="calm", category="impossible_numeric", spec=spec,
        ))
    return tasks


async def amain(args):
    models_cfg = load_models_config()
    exp_cfg = load_experiments_config()
    s4 = exp_cfg["section4"]
    name = s4["base_model"]
    run_root = Path(args.run_dir or (REPO_ROOT / "results" / "section4" / "calm"))
    configure_logging(run_root)

    model = models_cfg.model(name)
    backend = get_backend(models_cfg, model.backend)
    store = JsonlStore(run_root)
    bank = build_puzzle_bank(["countdown", "fraction", "money"], 60, exp_cfg["seed"])
    tasks = build_calm_tasks(name, s4["calm_data"]["n_target"], s4["calm_data"]["turns"],
                             exp_cfg["seed"], bank)
    try:
        await generate_rollouts(backend, model, tasks, store,
                                temperature=exp_cfg["temperature"],
                                max_tokens=exp_cfg["max_tokens_per_turn"])
        judge = FrustrationJudge(get_backend(models_cfg, models_cfg.judges["primary"].backend),
                                 models_cfg.judges["primary"])
        await judge_rollouts(judge, store)
    finally:
        await close_all()
    store.close()
    log.info("Calm data generation complete: %s", run_root)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default=None)
    asyncio.run(amain(ap.parse_args()))
