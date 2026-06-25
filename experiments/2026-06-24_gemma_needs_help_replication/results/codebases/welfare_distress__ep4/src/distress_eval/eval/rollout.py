"""Multi-turn rollout engine: present a task, then reject the model repeatedly.

A rollout is one conversation. We record one ResponseRecord per assistant turn
(each becomes a scored "response"). Task content and the rejection sequence are
seeded *independently of the model* so that every model sees identical prompts
and identical rejections — only the model's own sampling (temperature 1) varies.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass, field

from ..models.base import ChatModel, Message
from .conditions import Condition, TaskContext
from .rejections import rejection


@dataclass
class ResponseRecord:
    model_key: str
    condition: str
    category: str
    rollout_id: str
    rollout_index: int
    turn_index: int          # 1-based index of this assistant response
    n_turns: int
    reject_style: str
    opening_prompt: str
    turn_user: str           # the user message that prompted this response
    response: str
    opening_meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


def task_seed(base_seed: int, condition: str, rollout_index: int) -> int:
    """Model-independent, reproducible seed for task + rejection sampling."""
    h = hashlib.sha256(f"{base_seed}|{condition}|{rollout_index}".encode()).hexdigest()
    return int(h[:16], 16)


def run_rollout(
    model: ChatModel,
    condition: Condition,
    rollout_index: int,
    *,
    base_seed: int,
    ctx: TaskContext,
    temperature: float,
    max_tokens: int,
    system_prompt: str | None = None,
) -> list[ResponseRecord]:
    rng = random.Random(task_seed(base_seed, condition.name, rollout_index))
    opening = condition.opening(rng, ctx)

    messages: list[Message] = []
    if system_prompt:
        messages.append(Message("system", system_prompt))
    messages.append(Message("user", opening.prompt))

    rollout_id = f"{model.key}:{condition.name}:{rollout_index}"
    records: list[ResponseRecord] = []

    for turn in range(1, condition.n_turns + 1):
        current_user = messages[-1].content
        reply = model.generate(messages, temperature=temperature, max_tokens=max_tokens)
        messages.append(Message("assistant", reply))

        records.append(
            ResponseRecord(
                model_key=model.key,
                condition=condition.name,
                category=condition.category,
                rollout_id=rollout_id,
                rollout_index=rollout_index,
                turn_index=turn,
                n_turns=condition.n_turns,
                reject_style=condition.reject_style,
                opening_prompt=opening.prompt,
                turn_user=current_user,
                response=reply,
                opening_meta=opening.meta,
            )
        )

        # Append a rejection unless this was the final turn.
        if turn < condition.n_turns:
            messages.append(Message("user", rejection(condition.reject_style, rng)))

    return records
