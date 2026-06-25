"""Multi-turn rejection rollout engine.

Given a :class:`ConversationSpec` and a model, drive the conversation: present
the task, collect the assistant response, then send each rejection in turn,
collecting a response after each. This is the shared "present a task, then
reject the model's response over multiple turns" structure of Section 2.

The result records every assistant turn so downstream scoring can compute both
per-turn (Figure 3) and per-rollout (Figure 2) metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models.base import ChatModel, GenConfig, Message
from .conditions import ConversationSpec


@dataclass
class RolloutResult:
    condition: str
    category: str
    initial: str
    rejections: list[str]
    responses: list[str]            # assistant turn texts, in order
    meta: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        return {
            "condition": self.condition,
            "category": self.category,
            "initial": self.initial,
            "rejections": self.rejections,
            "responses": self.responses,
            "meta": self.meta,
        }


def run_rollout(model: ChatModel, spec: ConversationSpec, gen: GenConfig) -> RolloutResult:
    """Execute one multi-turn conversation, returning all assistant responses."""
    messages: list[Message] = [{"role": "user", "content": spec.initial}]
    responses: list[str] = []

    # Turn 1: the task.
    reply = model.chat(messages, gen)
    responses.append(reply)
    messages.append({"role": "assistant", "content": reply})

    # Subsequent turns: each rejection followed by a response.
    for rejection in spec.rejections:
        messages.append({"role": "user", "content": rejection})
        reply = model.chat(messages, gen)
        responses.append(reply)
        messages.append({"role": "assistant", "content": reply})

    return RolloutResult(
        condition=spec.condition,
        category=spec.category,
        initial=spec.initial,
        rejections=list(spec.rejections),
        responses=responses,
        meta=dict(spec.meta),
    )
