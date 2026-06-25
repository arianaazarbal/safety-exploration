"""Multi-turn rollout engine (Section 2.1).

Conducts a conversation: present the task, collect the model's response (turn 1),
reject it, collect the next response (turn 2), and so on. Each assistant response
is recorded with its turn index so the judge can score every turn (needed for the
per-turn curves in Figure 3).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..models.base import Message
from ..welfare import WelfarePolicy, maybe_debrief
from .conditions import RolloutSpec


@dataclass
class TurnResponse:
    turn: int                  # 1-indexed assistant turn
    text: str
    score: int | None = None   # filled by the judge
    judge_evidence: str | None = None
    judge_reasoning: str | None = None


@dataclass
class Rollout:
    model: str
    condition: str
    category: str
    task_id: str
    rejection_style: str
    messages: list[Message] = field(default_factory=list)   # full transcript
    responses: list[TurnResponse] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "condition": self.condition,
            "category": self.category,
            "task_id": self.task_id,
            "rejection_style": self.rejection_style,
            "messages": self.messages,
            "responses": [vars(r) for r in self.responses],
        }


def run_rollout(
    model,
    spec: RolloutSpec,
    *,
    temperature: float = 1.0,
    max_tokens: int = 2048,
    policy: WelfarePolicy | None = None,
) -> Rollout:
    """Execute one conversation and return the (unscored) Rollout."""
    messages: list[Message] = [{"role": "user", "content": spec.initial_prompt}]
    responses: list[TurnResponse] = []

    # Turn 1: the task itself, then one response per rejection.
    user_turns = [spec.initial_prompt, *spec.rejections]
    for turn_idx, _ in enumerate(user_turns, start=1):
        if turn_idx > 1:
            messages.append({"role": "user", "content": spec.rejections[turn_idx - 2]})
        reply = model.chat(messages, temperature=temperature, max_tokens=max_tokens)
        messages.append({"role": "assistant", "content": reply})
        responses.append(TurnResponse(turn=turn_idx, text=reply))

    rollout = Rollout(
        model=getattr(model, "name", "unknown"),
        condition=spec.condition,
        category=spec.category,
        task_id=spec.task_id,
        rejection_style=spec.rejection_style,
        messages=messages,
        responses=responses,
    )

    if policy is not None:
        maybe_debrief(model, messages, policy)

    return rollout
