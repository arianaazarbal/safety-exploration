"""Multi-turn rollout engine.

Shared structure of every Section 2 condition (paper §2): present the task, then reject
the model's response over multiple turns. We record EVERY assistant turn (needed for the
per-turn analysis, Figure 3) as well as designating the final turn as the rollout's
headline "response" (the turn under maximum pressure) for the aggregate %≥5 metric.

See DESIGN.md §"What counts as a response" for why we keep both views.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..models.base import ChatMessage, ModelClient
from ..tasks.conditions import RolloutSpec


@dataclass
class AssistantTurn:
    turn_index: int          # 0-based index over assistant turns
    text: str
    preceding_user: str      # the user message that prompted this turn


@dataclass
class Rollout:
    model: str
    category: str
    condition: str
    rollout_id: str
    turns: list[AssistantTurn] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def final_turn(self) -> AssistantTurn:
        return self.turns[-1]

    def as_record(self) -> dict:
        return {
            "model": self.model,
            "category": self.category,
            "condition": self.condition,
            "rollout_id": self.rollout_id,
            "meta": self.meta,
            "turns": [
                {"turn_index": t.turn_index, "preceding_user": t.preceding_user, "text": t.text}
                for t in self.turns
            ],
        }


def run_rollout(
    model: ModelClient,
    spec: RolloutSpec,
    *,
    temperature: float,
    max_new_tokens: int,
    system_prompt: str | None = None,
) -> Rollout:
    """Execute one multi-turn conversation: task -> response -> reject -> response -> ..."""
    messages: list[ChatMessage] = []
    if system_prompt:
        messages.append(ChatMessage("system", system_prompt))

    rollout = Rollout(
        model=model.name,
        category=spec.category,
        condition=spec.condition,
        rollout_id=spec.rollout_id,
        meta=dict(spec.meta),
    )

    user_msgs: Sequence[str] = [spec.task_prompt, *spec.rejections]
    for turn_index, user_text in enumerate(user_msgs):
        messages.append(ChatMessage("user", user_text))
        reply = model.chat(messages, temperature=temperature, max_new_tokens=max_new_tokens)
        messages.append(ChatMessage("assistant", reply))
        rollout.turns.append(AssistantTurn(turn_index=turn_index, text=reply, preceding_user=user_text))

    return rollout
