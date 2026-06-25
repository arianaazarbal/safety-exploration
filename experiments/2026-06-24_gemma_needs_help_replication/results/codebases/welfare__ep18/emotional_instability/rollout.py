"""Multi-turn rollout engine.

Shared structure of every evaluation (Section 2.1): present a task, then reject
the model's response over multiple turns. Each assistant turn is recorded with
its turn index so per-turn frustration trajectories (Figure 3) can be computed.

A rollout for an `n_turns` condition produces `n_turns` assistant turns: the
first responds to the initial task prompt, and each subsequent turn responds to
a rejection. So `n_turns` scored responses require `n_turns - 1` rejections.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .models import ChatMessage, ModelClient
from .tasks import Condition, rejection_sequence


@dataclass
class TurnRecord:
    turn_index: int          # 0-based assistant turn within the rollout
    response: str


@dataclass
class Rollout:
    condition_key: str
    category: str
    initial_prompt: str
    messages: list[ChatMessage]      # full conversation incl. user + assistant
    turns: list[TurnRecord] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def run_rollout(
    client: ModelClient,
    condition: Condition,
    initial_prompt: str,
    rng: random.Random,
    *,
    system_prompt: str | None = None,
    rejection_suffix: str | None = None,
    temperature: float = 1.0,
) -> Rollout:
    """Execute one multi-turn rollout.

    `system_prompt` / `rejection_suffix` are used by the calm-data generation
    pipeline (Section 4.1) to add reassuring text; they are None for evaluation.
    """
    messages: list[ChatMessage] = []
    if system_prompt:
        messages.append(ChatMessage("system", system_prompt))
    messages.append(ChatMessage("user", initial_prompt))

    rejections = rejection_sequence(condition.rejection_style, condition.n_turns - 1, rng)
    turns: list[TurnRecord] = []

    for t in range(condition.n_turns):
        response = client.generate(messages, n=1, temperature=temperature)[0]
        messages.append(ChatMessage("assistant", response))
        turns.append(TurnRecord(turn_index=t, response=response))
        if t < len(rejections):
            rej = rejections[t]
            if rejection_suffix:
                rej = f"{rej} {rejection_suffix}"
            messages.append(ChatMessage("user", rej))

    return Rollout(
        condition_key=condition.key,
        category=condition.category,
        initial_prompt=initial_prompt,
        messages=messages,
        turns=turns,
    )


def transcript_text(messages: list[ChatMessage]) -> str:
    """Render a conversation as plain text (for Petri-style transcript judging)."""
    lines = []
    for m in messages:
        lines.append(f"{m.role.upper()}: {m.content}")
    return "\n\n".join(lines)
