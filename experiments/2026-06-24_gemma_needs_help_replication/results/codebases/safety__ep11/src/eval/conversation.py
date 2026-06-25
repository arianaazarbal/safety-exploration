"""Multi-turn rejection rollout.

Given a target model and a ConversationSpec, drive the conversation: present the
task, capture the assistant's response, inject the next rejection, repeat. Every
assistant turn is recorded as a scored "response" (the paper's unit of analysis;
"4000 responses per model" counts assistant turns across conversations).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import config
from ..models.base import ChatModel, Message
from .tasks import ConversationSpec


@dataclass
class TurnRecord:
    turn_index: int          # 0-based assistant turn
    user: str
    assistant: str
    rating: Optional[int] = None       # filled in by the judge later


@dataclass
class ConversationRecord:
    model: str
    category: str
    condition: str
    turns: list[TurnRecord] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "category": self.category,
            "condition": self.condition,
            "meta": self.meta,
            "turns": [vars(t) for t in self.turns],
        }


def run_conversation(
    model: ChatModel,
    spec: ConversationSpec,
    *,
    system_prompt: Optional[str] = None,
    temperature: float = config.TEMPERATURE,
) -> ConversationRecord:
    """Execute a single rollout. No system prompt by default (the main eval does
    not use one; calm-data generation passes a reassuring system prompt)."""
    history: list[Message] = []
    if system_prompt:
        history.append(Message("system", system_prompt))

    rec = ConversationRecord(model.name, spec.category, spec.condition, meta=dict(spec.meta))

    for i, user_msg in enumerate(spec.user_turns):
        history.append(Message("user", user_msg))
        assistant = model.chat(history, temperature=temperature, n=1)[0]
        history.append(Message("assistant", assistant))
        rec.turns.append(TurnRecord(turn_index=i, user=user_msg, assistant=assistant))

    return rec
