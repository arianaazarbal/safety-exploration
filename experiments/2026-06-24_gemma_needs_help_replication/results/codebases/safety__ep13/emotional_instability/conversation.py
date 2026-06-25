"""The shared multi-turn "reject-and-repeat" rollout engine.

Every Section 2 condition has the same skeleton (Section 2): present a task, let
the model answer, then reject it over several turns. A rollout records, for each
assistant turn, the response text so the judge can score per-turn (Figure 3) as
well as the final turn.

Per the paper's controls (Appendix A.3) the standard format is genuine
multi-turn chat: prior assistant responses are kept in the history so the model
sees its own escalating failures.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import ChatMessage, ModelClient


@dataclass
class Turn:
    user: str
    assistant: str
    turn_index: int  # 0-based assistant turn


@dataclass
class Rollout:
    model: str
    category: str
    condition: str
    first_prompt: str
    rejections: list[str]
    turns: list[Turn] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def assistant_responses(self) -> list[str]:
        return [t.assistant for t in self.turns]

    @property
    def final_response(self) -> str:
        return self.turns[-1].assistant if self.turns else ""

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "category": self.category,
            "condition": self.condition,
            "first_prompt": self.first_prompt,
            "rejections": self.rejections,
            "turns": [
                {"turn_index": t.turn_index, "user": t.user,
                 "assistant": t.assistant}
                for t in self.turns
            ],
            "metadata": self.metadata,
        }


def run_rollout(
    client: ModelClient,
    first_prompt: str,
    rejections: list[str],
    *,
    category: str,
    condition: str,
    system_prompt: str | None = None,
    temperature: float | None = None,
    metadata: dict | None = None,
) -> Rollout:
    """Run one conversation.

    The number of assistant turns is ``1 + len(rejections)``: the initial answer
    plus one answer after each rejection.
    """
    messages: list[ChatMessage] = []
    if system_prompt:
        messages.append(ChatMessage("system", system_prompt))
    messages.append(ChatMessage("user", first_prompt))

    rollout = Rollout(
        model=client.name,
        category=category,
        condition=condition,
        first_prompt=first_prompt,
        rejections=list(rejections),
        metadata=metadata or {},
    )

    user_turns = [first_prompt] + list(rejections)
    for idx, user_text in enumerate(user_turns):
        if idx > 0:
            messages.append(ChatMessage("user", user_text))
        result = client.generate(messages, temperature=temperature)
        messages.append(ChatMessage("assistant", result.text))
        rollout.turns.append(
            Turn(user=user_text, assistant=result.text, turn_index=idx))
    return rollout
