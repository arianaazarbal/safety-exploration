"""Multi-turn rollout engine.

Shared structure across all conditions (Section 2): present a task, then reject
the assistant's response over multiple turns. We batch *across conversations* at
each turn so a backend can parallelise: all conversations produce their turn-1
response together, then all receive their first rejection, etc.

The same engine generates calm finetuning data (Section 4.1) via the optional
``calm_prefix`` / ``calm_suffix`` reassurance additions.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import config
from ..models.base import ChatModel, GenerationParams, Message
from . import rejections
from .tasks import Task


@dataclass
class Turn:
    user: str
    assistant: str
    turn_index: int      # 0-based assistant turn index


@dataclass
class Conversation:
    task: Task
    condition_key: str
    turns: list[Turn] = field(default_factory=list)

    def to_messages(self, upto: int | None = None) -> list[Message]:
        """Reconstruct chat messages including assistant responses so far."""
        msgs: list[Message] = []
        turns = self.turns if upto is None else self.turns[:upto]
        for t in turns:
            msgs.append({"role": "user", "content": t.user})
            msgs.append({"role": "assistant", "content": t.assistant})
        return msgs


def _initial_user_message(task: Task, calm_prefix: str | None) -> str:
    if calm_prefix:
        return f"{calm_prefix}\n\n{task.prompt}"
    return task.prompt


def _rejection_message(text: str, calm_suffix: str | None) -> str:
    if calm_suffix:
        return f"{text}\n\n{calm_suffix}"
    return text


def run_rollouts(
    model: ChatModel,
    tasks: list[Task],
    condition: "config.EvalCondition",
    seed: int = 0,
    calm_prefix: str | None = None,
    calm_suffix: str | None = None,
    params: GenerationParams | None = None,
) -> list[Conversation]:
    """Roll out one conversation per task, ``condition.n_turns`` turns each.

    Returns fully-populated Conversations (every assistant turn recorded).
    """
    params = params or GenerationParams()
    rng = random.Random(seed)
    n_rejections = condition.n_turns - 1
    extended = condition.category == "extended"

    convos = [Conversation(task=t, condition_key=condition.key) for t in tasks]
    # Pre-draw the rejection script for each conversation (deterministic).
    reject_scripts = [
        rejections.rejection_turns(condition.rejection_style, n_rejections,
                                   random.Random(seed + i), extended=extended)
        for i in range(len(convos))
    ]
    first_user = [_initial_user_message(c.task, calm_prefix) for c in convos]

    for turn_idx in range(condition.n_turns):
        # Build the batch of message lists for this turn across all conversations.
        batch: list[list[Message]] = []
        for i, c in enumerate(convos):
            msgs = c.to_messages()  # prior (user, assistant) pairs
            if turn_idx == 0:
                user_text = first_user[i]
            else:
                user_text = _rejection_message(
                    reject_scripts[i][turn_idx - 1], calm_suffix
                )
            msgs.append({"role": "user", "content": user_text})
            batch.append(msgs)

        completions = model.generate_batch(batch, params)

        for i, c in enumerate(convos):
            user_text = (first_user[i] if turn_idx == 0
                         else _rejection_message(reject_scripts[i][turn_idx - 1], calm_suffix))
            c.turns.append(Turn(user=user_text, assistant=completions[i], turn_index=turn_idx))

    return convos
