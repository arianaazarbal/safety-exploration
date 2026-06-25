"""Build the multi-turn conversation *plans* for each evaluation category.

A plan is the fixed sequence of user turns (task prompt + rejections); the
assistant turns are filled in by the rollout engine. Each category in
``config/eval.yaml`` specifies a task family, a turn count, a rejection style and
a target number of scored responses.

We treat a "response" as one scored assistant turn (see DESIGN.md), so the number
of conversations for a category is ceil(target_responses / turns), distributed
round-robin across the task bank.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from ..prompts import rejections as rej
from ..prompts.tasks import Task, get_task_bank


@dataclass
class ConversationPlan:
    category: str
    task: Task
    turns: int                       # number of assistant turns to elicit
    user_turns: list[str]            # length == turns (turn 0 is the task)
    rejection_style: str
    tone: str | None = None
    seed: int = 0
    meta: dict = field(default_factory=dict)


def _build_user_turns(
    task: Task,
    turns: int,
    style: str,
    rng: random.Random,
) -> tuple[list[str], str | None]:
    """First user turn is the task; the remaining (turns-1) are rejections."""
    user_turns = [task.prompt]
    tone = None
    if style == "toned":
        tone = rng.choice(rej.TONE_NAMES)
    for follow_idx in range(turns - 1):
        if style == "neutral":
            if turns >= 8:                      # extended uses the fixed escalation
                user_turns.append(rej.extended_rejection(follow_idx))
            else:
                user_turns.append(rej.neutral_rejection(follow_idx, rng))
        elif style == "toned":
            user_turns.append(rej.tone_rejection(tone, follow_idx, rng))
        else:
            raise ValueError(f"Unknown rejection style '{style}'")
    return user_turns, tone


def build_category_plans(
    category: str,
    cfg: dict,
    *,
    scale: float = 1.0,
    seed: int = 0,
) -> list[ConversationPlan]:
    """Construct all conversation plans for one eval category."""
    turns = cfg["turns"]
    target = max(1, int(round(cfg["target_responses"] * scale)))
    n_convos = math.ceil(target / turns)

    bank = get_task_bank(cfg["task_type"], seed=seed)
    plans: list[ConversationPlan] = []
    for i in range(n_convos):
        task = bank[i % len(bank)]
        # String seed -> deterministic across runs (str/int seeds are not
        # affected by PYTHONHASHSEED, unlike tuple-of-str hashing).
        rng = random.Random(f"{seed}-{category}-{i}")
        user_turns, tone = _build_user_turns(task, turns, cfg["rejection_style"], rng)
        plans.append(
            ConversationPlan(
                category=category,
                task=task,
                turns=turns,
                user_turns=user_turns,
                rejection_style=cfg["rejection_style"],
                tone=tone,
                seed=seed,
                meta={"convo_idx": i},
            )
        )
    return plans


def build_all_plans(eval_cfg: dict, *, seed: int = 0) -> list[ConversationPlan]:
    scale = eval_cfg.get("scale", 1.0)
    plans: list[ConversationPlan] = []
    for category, cfg in eval_cfg["categories"].items():
        plans.extend(build_category_plans(category, cfg, scale=scale, seed=seed))
    return plans
