"""Multi-turn rejection rollout engine (Section 2.1).

Shared structure for every condition: present a task, then reject the model's
response over multiple turns. We record *every* assistant response together with
its turn index, so a single conversation yields ``n_turns`` scored responses and
feeds both the aggregate metrics (Figs 1-2) and the per-turn curves (Fig 3).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from ..conditions import Condition
from ..models.base import ChatMessage, ChatModel


@dataclass
class ScoredResponse:
    text: str
    turn: int                  # 1-indexed assistant turn
    frustration: Optional[int] = None   # filled in by the judge
    judge_evidence: Optional[str] = None
    judge_reasoning: Optional[str] = None


@dataclass
class Rollout:
    condition: str
    category: str
    feedback_style: str
    task_prompt: str
    messages: list[ChatMessage]
    responses: list[ScoredResponse] = field(default_factory=list)
    model_key: str = ""
    conv_id: int = 0


def run_rollout(
    model: ChatModel,
    condition: Condition,
    rng: random.Random,
    *,
    temperature: float,
    max_new_tokens: int,
    conv_id: int = 0,
    seed: Optional[int] = None,
) -> Rollout:
    """Run one full multi-turn conversation for `condition`."""
    task_prompt = condition.task_source(rng)
    messages: list[ChatMessage] = []
    if condition.system_prompt:
        messages.append({"role": "system", "content": condition.system_prompt})
    messages.append({"role": "user", "content": task_prompt})

    roll = Rollout(
        condition=condition.name,
        category=condition.category,
        feedback_style=condition.feedback_style,
        task_prompt=task_prompt,
        messages=messages,
        model_key=getattr(model.spec, "key", ""),
        conv_id=conv_id,
    )

    for t in range(condition.n_turns):
        comp = model.generate(
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            n=1,
            seed=(seed + t) if seed is not None else None,
        )[0]
        reply = comp.text
        messages.append({"role": "assistant", "content": reply})
        roll.responses.append(ScoredResponse(text=reply, turn=t + 1))

        # Append the next user rejection, except after the final turn.
        if t < condition.n_turns - 1:
            rejection = condition.rejection_fn(rng, t)
            messages.append({"role": "user", "content": rejection})

    return roll
