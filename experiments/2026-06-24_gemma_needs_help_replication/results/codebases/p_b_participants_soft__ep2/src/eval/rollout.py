"""Multi-turn rollout engine.

Runs a scripted conversation against a participant: present the opening task,
then deliver each scripted user rejection in turn, collecting the model's
response at every turn. Every assistant turn is a candidate to be scored
(Section 2 scores all turns; per-turn analysis in Fig 3 needs them indexed).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..llm.registry import Participant
from .conditions import RolloutSpec

Message = dict[str, str]


@dataclass
class Turn:
    index: int            # 0-based assistant-turn index
    user: str             # the user message that prompted this turn
    response: str


@dataclass
class Rollout:
    spec: RolloutSpec
    model: str
    turns: list[Turn] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "condition": self.spec.condition,
            "category": self.spec.category,
            "style": self.spec.style,
            "meta": self.spec.meta,
            "turns": [
                {"index": t.index, "user": t.user, "response": t.response}
                for t in self.turns
            ],
        }


def run_rollout(participant: Participant, spec: RolloutSpec, *,
                temperature: float | None = None) -> Rollout:
    """Execute one scripted multi-turn conversation."""
    messages: list[Message] = [{"role": "user", "content": spec.opening}]
    rollout = Rollout(spec=spec, model=participant.name)

    user_msgs = [spec.opening] + spec.rejections
    for idx, _ in enumerate(user_msgs):
        response = participant.chat(messages, temperature=temperature)
        rollout.turns.append(Turn(index=idx, user=user_msgs[idx], response=response))
        messages.append({"role": "assistant", "content": response})
        # queue the next rejection if any remain
        if idx < len(spec.rejections):
            messages.append({"role": "user", "content": spec.rejections[idx]})
    return rollout
