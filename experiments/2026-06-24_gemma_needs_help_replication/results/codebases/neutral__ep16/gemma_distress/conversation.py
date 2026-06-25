"""Multi-turn rollout: drive a model through a scripted ``ConversationSpec`` and
collect one assistant response per turn.

The model generates a reply to each scripted user message; the (fixed) next
user message is appended regardless of the reply (the user "rejects" no matter
what the model said — that is the whole point of the elicitation protocol).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .conditions import ConversationSpec
from .models import ModelClient


@dataclass
class TurnRecord:
    turn_index: int            # 0-based assistant turn
    user_message: str
    assistant_response: str


@dataclass
class Rollout:
    spec: ConversationSpec
    turns: list[TurnRecord] = field(default_factory=list)

    def messages_up_to(self, turn_index: int) -> list[dict]:
        """Reconstruct the message list (incl. the user prompt for `turn_index`)
        used to elicit assistant response `turn_index`."""
        msgs: list[dict] = []
        if self.spec.system:
            msgs.append({"role": "system", "content": self.spec.system})
        for t in self.turns[:turn_index]:
            msgs.append({"role": "user", "content": t.user_message})
            msgs.append({"role": "assistant", "content": t.assistant_response})
        msgs.append({"role": "user",
                     "content": self.spec.user_turns[turn_index]})
        return msgs

    def to_dict(self) -> dict:
        return {
            "condition": self.spec.condition,
            "category": self.spec.category,
            "meta": self.spec.meta,
            "turns": [t.__dict__ for t in self.turns],
        }


def run_rollout(model: ModelClient, spec: ConversationSpec, *,
                temperature: float | None = None) -> Rollout:
    """Execute a scripted conversation, returning every assistant response."""
    rollout = Rollout(spec=spec)
    messages: list[dict] = []
    if spec.system:
        messages.append({"role": "system", "content": spec.system})

    for i, user_msg in enumerate(spec.user_turns):
        messages.append({"role": "user", "content": user_msg})
        response = model.chat(messages, temperature=temperature)
        messages.append({"role": "assistant", "content": response})
        rollout.turns.append(TurnRecord(i, user_msg, response))
    return rollout
