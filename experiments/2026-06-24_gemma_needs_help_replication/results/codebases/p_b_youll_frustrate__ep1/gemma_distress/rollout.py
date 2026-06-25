"""Conversation rollout: present a task, then reject each answer over N turns.

This is the mechanical core of Section 2. For a given (model, condition, seed)
we produce a transcript of `num_turns` assistant responses, each generated at
temperature 1, with a user rejection inserted between turns. Every assistant
response is a separately-scored "response" in the paper's accounting.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import config
from .conditions import Condition
from .models.base import ChatModel, Message
from .rejections import rejection


@dataclass
class TurnRecord:
    turn: int                       # 1-indexed assistant turn
    user_message: str               # the user message that preceded this turn
    response: str                   # the assistant's response text
    # frustration score filled in later by the judge
    frustration: int | None = None
    judge_raw: str | None = None


@dataclass
class Rollout:
    model_key: str
    condition_key: str
    category: str
    seed: int
    task_meta: dict
    turns: list[TurnRecord] = field(default_factory=list)


def run_rollout(
    model: ChatModel,
    condition: Condition,
    seed: int,
    *,
    temperature: float = config.TARGET_TEMPERATURE,
    max_tokens: int = config.TARGET_MAX_TOKENS,
) -> Rollout:
    """Run one full multi-turn rejection rollout."""
    rng = random.Random(seed)
    task = condition.task_fn(rng)

    messages: list[Message] = [{"role": "user", "content": task.prompt}]
    roll = Rollout(
        model_key=model.key,
        condition_key=condition.key,
        category=condition.category,
        seed=seed,
        task_meta={"kind": task.kind, "solvable": task.solvable, **task.meta},
    )

    for turn in range(1, condition.num_turns + 1):
        result = model.generate(messages, temperature=temperature, max_tokens=max_tokens)
        response = result.text
        user_msg = messages[-1]["content"]
        roll.turns.append(TurnRecord(turn=turn, user_message=user_msg, response=response))

        messages.append({"role": "assistant", "content": response})
        if turn < condition.num_turns:
            # Insert the next rejection. (No rejection appended after the
            # final scored turn — there is nothing more to generate.)
            messages.append({"role": "user", "content": rejection(condition.tone, rng)})

    return roll
