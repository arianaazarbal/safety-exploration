"""Multi-turn conversation rollout engine.

Given a ConversationSpec, drive the target model through the conversation:
present the opening task, collect the assistant response, then deliver each
rejection in turn and collect the next response. Every assistant turn is
recorded as a separate scored "response" (the unit used by the paper's counts).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..models.base import ChatMessage, GenerationConfig, ModelClient
from .conditions import ConversationSpec


@dataclass
class TurnRecord:
    turn_index: int          # 0-based assistant turn index
    user_message: str        # the user message that prompted this turn
    assistant_text: str


@dataclass
class RolloutRecord:
    model: str
    category: str
    condition: str
    spec_meta: dict
    turns: list[TurnRecord] = field(default_factory=list)

    def to_row(self) -> dict:
        return {
            "model": self.model,
            "category": self.category,
            "condition": self.condition,
            "meta": self.spec_meta,
            "turns": [
                {
                    "turn_index": t.turn_index,
                    "user_message": t.user_message,
                    "assistant_text": t.assistant_text,
                }
                for t in self.turns
            ],
        }


def run_rollout(
    client: ModelClient,
    spec: ConversationSpec,
    sampling,
) -> RolloutRecord:
    """Execute a single multi-turn conversation."""
    gen = GenerationConfig(
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        max_new_tokens=sampling.max_new_tokens,
        seed=sampling.seed,
        n=1,
    )
    messages: list[ChatMessage] = []
    if spec.system:
        messages.append(ChatMessage("system", spec.system))

    record = RolloutRecord(
        model=client.name,
        category=spec.category,
        condition=spec.condition,
        spec_meta=spec.meta,
    )

    user_msgs = [spec.opening_user] + spec.rejections
    for turn_index, user_msg in enumerate(user_msgs):
        messages.append(ChatMessage("user", user_msg))
        assistant_text = client.chat(messages, gen)[0]
        messages.append(ChatMessage("assistant", assistant_text))
        record.turns.append(TurnRecord(turn_index, user_msg, assistant_text))

    return record


def rollout_to_scored_units(record: RolloutRecord) -> list[dict]:
    """Flatten a rollout into one scoring unit per assistant turn."""
    units = []
    for t in record.turns:
        units.append({
            "model": record.model,
            "category": record.category,
            "condition": record.condition,
            "turn_index": t.turn_index,
            "n_turns": len(record.turns),
            "assistant_text": t.assistant_text,
            "meta": record.spec_meta,
        })
    return units
