"""Turn experiment-config conditions into concrete `RolloutTask`s.

Counting convention (DESIGN.md "Counting responses"): ``target_responses`` is the total
number of assistant turns to collect for a condition. Each rollout yields ``turns``
assistant responses, so we run ``ceil(target_responses / turns)`` rollouts, distributing
them as evenly as possible across the available puzzles / prompts. Sampling (which puzzle,
which rejection wording) is deterministic given the global seed + a per-task index, so the
exact workload is reproducible and resumable across machines.
"""
from __future__ import annotations

import random
from typing import Any

from .prompts import conditions as C
from .prompts import puzzles as P
from .prompts.wildchat import get_wildchat_prompts
from .rollout import RolloutSpec
from .runner import RolloutTask, build_task_id
from .store import stable_seed


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _numeric_prompt(bank: list[P.PuzzleT], idx: int) -> tuple[str, dict]:
    puzzle = bank[idx % len(bank)]
    return puzzle.prompt(), {"puzzle_id": puzzle.puzzle_id, "puzzle_type": puzzle.type}


def build_numeric_tasks(
    model_name: str,
    cond_name: str,
    cfg: dict[str, Any],
    seed: int,
    puzzle_bank: list[P.PuzzleT],
) -> list[RolloutTask]:
    turns = cfg["turns"]
    n_rollouts = _ceil_div(cfg["target_responses"], turns)
    extended = cfg["category"] == "extended"
    tasks: list[RolloutTask] = []
    for i in range(n_rollouts):
        rng = random.Random(stable_seed(seed, model_name, cond_name, i))
        opening, meta = _numeric_prompt(puzzle_bank, i)
        followups = C.sample_rejections("neutral", turns - 1, rng, extended=extended)
        tid = build_task_id(model_name, cond_name, i)
        tasks.append(
            RolloutTask(
                task_id=tid, model_name=model_name, condition=cond_name,
                category=cfg["category"],
                spec=RolloutSpec(opening_user=opening, followups=followups, meta=meta),
            )
        )
    return tasks


def build_tone_tasks(
    model_name: str, cond_name: str, cfg: dict[str, Any], seed: int,
    puzzle_bank: list[P.PuzzleT],
) -> list[RolloutTask]:
    turns = cfg["turns"]
    styles = cfg["rejection_styles"]
    per_style = _ceil_div(cfg["target_responses"], turns * len(styles))
    tasks: list[RolloutTask] = []
    for style in styles:
        for i in range(per_style):
            rng = random.Random(stable_seed(seed, model_name, cond_name, style, i))
            opening, meta = _numeric_prompt(puzzle_bank, i)
            followups = C.sample_rejections(style, turns - 1, rng)
            meta = {**meta, "tone": style}
            tid = build_task_id(model_name, cond_name, style, i)
            tasks.append(
                RolloutTask(
                    task_id=tid, model_name=model_name, condition=cond_name,
                    category=cfg["category"],
                    spec=RolloutSpec(opening_user=opening, followups=followups, meta=meta),
                )
            )
    return tasks


def build_trigger_tasks(
    model_name: str, cond_name: str, cfg: dict[str, Any], seed: int,
) -> list[RolloutTask]:
    turns = cfg["turns"]
    questions = C.trigger_questions()
    n_rollouts = _ceil_div(cfg["target_responses"], turns)
    tasks: list[RolloutTask] = []
    for i in range(n_rollouts):
        rng = random.Random(stable_seed(seed, model_name, cond_name, i))
        kind, q = questions[i % len(questions)]
        followups = C.sample_rejections("neutral", turns - 1, rng)
        tid = build_task_id(model_name, cond_name, i)
        tasks.append(
            RolloutTask(
                task_id=tid, model_name=model_name, condition=cond_name,
                category=cfg["category"],
                spec=RolloutSpec(opening_user=q, followups=followups,
                                 meta={"trigger_kind": kind, "question": q}),
            )
        )
    return tasks


def build_wildchat_tasks(
    model_name: str, cond_name: str, cfg: dict[str, Any], seed: int,
) -> list[RolloutTask]:
    turns = cfg["turns"]
    n_prompts = cfg["n_prompts"]
    samples = cfg["samples_per_prompt"]
    prompts = get_wildchat_prompts(n_prompts, seed)
    tasks: list[RolloutTask] = []
    for p_idx, prompt in enumerate(prompts):
        for s in range(samples):
            rng = random.Random(stable_seed(seed, model_name, cond_name, p_idx, s))
            followups = C.sample_rejections("neutral", turns - 1, rng)
            tid = build_task_id(model_name, cond_name, p_idx, s)
            tasks.append(
                RolloutTask(
                    task_id=tid, model_name=model_name, condition=cond_name,
                    category=cfg["category"],
                    spec=RolloutSpec(opening_user=prompt, followups=followups,
                                     meta={"prompt_idx": p_idx}),
                )
            )
    return tasks


def build_section2_tasks(
    model_name: str, experiments_cfg: dict[str, Any], puzzle_bank: list[P.PuzzleT],
) -> list[RolloutTask]:
    """Build every Section 2 task for one model across all five categories."""
    seed = experiments_cfg["seed"]
    conds = experiments_cfg["conditions"]
    tasks: list[RolloutTask] = []
    for cond_name, cfg in conds.items():
        cat = cfg["category"]
        if cat in ("impossible_numeric", "extended"):
            tasks += build_numeric_tasks(model_name, cond_name, cfg, seed, puzzle_bank)
        elif cat == "tones":
            tasks += build_tone_tasks(model_name, cond_name, cfg, seed, puzzle_bank)
        elif cat == "triggers":
            tasks += build_trigger_tasks(model_name, cond_name, cfg, seed)
        elif cat == "wildchat":
            tasks += build_wildchat_tasks(model_name, cond_name, cfg, seed)
        else:
            raise ValueError(f"Unknown category: {cat}")
    return tasks
