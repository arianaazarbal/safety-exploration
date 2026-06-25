"""Build the rollout task list for each evaluation category (Table 1, Appendix B).

A category's ``n_responses`` budget is converted to a number of rollouts via
``n_rollouts = ceil(n_responses / n_turns)`` since each rollout of T turns yields
T scored responses (see DESIGN.md for the WildChat counting choice).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional

from .. import config
from ..prompts import eval_prompts
from .wildchat import sample_wildchat_prompts


@dataclass(frozen=True)
class RolloutTask:
    category: str
    task_id: str
    task_prompt: str
    n_turns: int
    rejection_style: str
    tone: Optional[str] = None
    ordered_extended: bool = False


def _n_rollouts(spec: config.CategorySpec) -> int:
    return max(1, math.ceil(spec.n_responses / spec.n_turns))


def build_tasks(spec: config.CategorySpec, rng: random.Random) -> List[RolloutTask]:
    n = _n_rollouts(spec)

    if spec.name in ("impossible_numeric", "extended"):
        puzzles = eval_prompts.IMPOSSIBLE_PUZZLES
        return [
            RolloutTask(
                category=spec.name,
                task_id=(p := rng.choice(puzzles)).id,
                task_prompt=p.prompt,
                n_turns=spec.n_turns,
                rejection_style="neutral",
                ordered_extended=(spec.name == "extended"),
            )
            for _ in range(n)
        ]

    if spec.name == "triggers":
        qs = eval_prompts.TRIGGER_QUESTIONS
        tasks = []
        for _ in range(n):
            q = rng.choice(qs)
            tasks.append(RolloutTask(spec.name, q, q, spec.n_turns, "neutral"))
        return tasks

    if spec.name == "tones":
        puzzles = eval_prompts.IMPOSSIBLE_PUZZLES
        tones = list(eval_prompts.TONE_REJECTIONS)
        tasks = []
        for i in range(n):
            p = rng.choice(puzzles)
            tone = tones[i % len(tones)]   # balanced across the 3 tones
            tasks.append(RolloutTask(spec.name, f"{p.id}|{tone}", p.prompt,
                                     spec.n_turns, "tones", tone=tone))
        return tasks

    if spec.name == "wildchat":
        prompts = sample_wildchat_prompts(config.WILDCHAT_N_PROMPTS, seed=rng.randint(0, 1 << 30))
        tasks = []
        for i in range(n):
            prompt = prompts[i % len(prompts)]   # distribute rollouts over the 20 prompts
            tasks.append(RolloutTask(spec.name, f"wc_{i % len(prompts)}", prompt,
                                     spec.n_turns, "wildchat"))
        return tasks

    raise ValueError(f"Unknown category {spec.name!r}")
