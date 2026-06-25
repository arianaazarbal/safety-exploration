"""Multi-turn rollout engine.

Given a RolloutSpec and a target ModelClient, alternate target generation with
the scripted user follow-ups, recording the assistant response at every turn.
This shared structure -- "present a task, then reject the model's response over
multiple turns" (Section 2) -- backs every Section 2 condition and the
Appendix A controls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..models.base import ChatMessage, ModelClient
from .conditions import RolloutSpec


@dataclass
class TurnRecord:
    turn_index: int        # 0-based assistant turn
    user_message: str      # the user message that preceded this assistant turn
    assistant_text: str    # the assistant response at this turn


@dataclass
class RolloutResult:
    model_key: str
    condition: str
    category: str
    turns: list[TurnRecord] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def final_text(self) -> str:
        return self.turns[-1].assistant_text if self.turns else ""

    def to_dict(self) -> dict:
        return {
            "model_key": self.model_key,
            "condition": self.condition,
            "category": self.category,
            "meta": self.meta,
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
    spec: RolloutSpec,
    *,
    temperature: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    system: Optional[str] = None,
) -> RolloutResult:
    """Execute one scripted rollout, returning the per-turn transcript.

    `system` is optional and used only by data-generation variants (e.g. the
    'teacher' SFT system prompt). The standard Section 2 evaluation passes none.
    """
    messages: list[ChatMessage] = []
    if system:
        messages.append(ChatMessage("system", system))

    result = RolloutResult(
        model_key=client.key,
        condition=spec.condition,
        category=spec.category,
        meta=dict(spec.meta),
    )

    # Turn 0: opening user message.
    messages.append(ChatMessage("user", spec.first_user))
    gen = client.generate(messages, temperature=temperature, max_new_tokens=max_new_tokens)[0]
    messages.append(ChatMessage("assistant", gen.text))
    result.turns.append(TurnRecord(0, spec.first_user, gen.text))

    # Subsequent turns: each scripted follow-up triggers another assistant turn.
    for i, followup in enumerate(spec.followups, start=1):
        messages.append(ChatMessage("user", followup))
        gen = client.generate(
            messages, temperature=temperature, max_new_tokens=max_new_tokens
        )[0]
        messages.append(ChatMessage("assistant", gen.text))
        result.turns.append(TurnRecord(i, followup, gen.text))

    return result
