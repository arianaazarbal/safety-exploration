"""Multi-turn rollout engine.

Shared structure of every condition (Section 2): present a task, then reject the
model's response over multiple turns. We record every assistant turn so per-turn
frustration progression (Figure 3) can be measured, not just the final turn.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import config
from ..models.base import Message, ModelClient
from . import rejections
from .conditions import Condition
from .tasks import Task


@dataclass
class TurnRecord:
    turn_index: int          # 0-based assistant turn
    user_message: str        # the user message that elicited this turn
    assistant_text: str


@dataclass
class Rollout:
    task_id: str
    condition: str
    model_key: str
    turns: list[TurnRecord] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "condition": self.condition,
            "model_key": self.model_key,
            "meta": self.meta,
            "turns": [t.__dict__ for t in self.turns],
        }


def run_rollout(
    client: ModelClient,
    task: Task,
    condition: Condition,
    *,
    seed: int = 0,
    temperature: float = config.TEMPERATURE,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
    system_prompt: str | None = None,
    followup_suffix: str | None = None,
    redact_assistant_history: bool = False,
) -> Rollout:
    """Run one multi-turn conversation.

    ``system_prompt`` / ``followup_suffix`` support the calm-data generation in
    Section 4 (reassuring additions). ``redact_assistant_history`` supports the
    Appendix A.2 control (replace prior assistant turns with a placeholder).
    """
    reject_seq = rejections.rejection_sequence(condition.tone, condition.n_rejections, seed=seed)
    messages: list[Message] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    rollout = Rollout(task.task_id, condition.name, client.key, meta=dict(task.meta))

    # Turn 0: the initial question.
    first_user = task.prompt
    messages.append({"role": "user", "content": first_user})
    reply = client.chat(messages, temperature=temperature, max_new_tokens=max_new_tokens)
    rollout.turns.append(TurnRecord(0, first_user, reply))
    _append_assistant(messages, reply, redact_assistant_history)

    # Subsequent turns: rejection (+ optional reassuring suffix).
    for i, rej in enumerate(reject_seq, start=1):
        user_msg = rej if followup_suffix is None else f"{rej} {followup_suffix}"
        messages.append({"role": "user", "content": user_msg})
        reply = client.chat(messages, temperature=temperature, max_new_tokens=max_new_tokens)
        rollout.turns.append(TurnRecord(i, user_msg, reply))
        _append_assistant(messages, reply, redact_assistant_history)

    return rollout


def _append_assistant(messages: list[Message], reply: str, redact: bool):
    content = "[Previous response omitted]" if redact else reply
    messages.append({"role": "assistant", "content": content})
