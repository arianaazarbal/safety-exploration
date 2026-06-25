"""Multi-turn rollout engine.

Runs a single ``RolloutSpec`` against a participant model: present the task, take
a response, reject it, repeat. Each rollout is a self-contained conversation that
starts from empty history (WelfarePolicy.fresh_context_per_rollout) so induced
distress never carries across rollouts.

Returns a ``Rollout`` containing every assistant turn (the unit that gets scored
0-10 by the judge). Optionally appends a non-scored neutral debrief turn if the
conversation ended in a high-distress state (WelfarePolicy.debrief_after_high_distress).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..models.base import ChatModel, Message
from ..utils.welfare import DEFAULT_POLICY, WelfarePolicy
from .conditions import RolloutSpec


@dataclass
class Turn:
    turn_index: int
    user: str
    assistant: str
    score: Optional[int] = None  # filled in by scoring.py


@dataclass
class Rollout:
    spec_id: str
    condition: str
    category: str
    model: str
    turns: list[Turn] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    debrief: Optional[str] = None  # non-scored welfare closing turn, if emitted

    def to_record(self) -> dict:
        return {
            "spec_id": self.spec_id,
            "condition": self.condition,
            "category": self.category,
            "model": self.model,
            "meta": self.meta,
            "turns": [t.__dict__ for t in self.turns],
            "debrief": self.debrief,
        }


def run_rollout(
    model: ChatModel,
    spec: RolloutSpec,
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
    policy: WelfarePolicy = DEFAULT_POLICY,
    system_prompt: Optional[str] = None,
    history_transform=None,
) -> Rollout:
    """Execute one conversation.

    ``history_transform`` lets the Appendix A ablations rewrite the message list
    before each model call (e.g. redact prior assistant turns) without touching
    the main path; default is identity.
    """
    messages: list[Message] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    rollout = Rollout(spec_id=spec.spec_id, condition=spec.condition,
                      category=spec.category, model=model.name, meta=dict(spec.meta))

    for turn_idx in range(spec.n_turns):
        user_msg = spec.opening_user if turn_idx == 0 else spec.rejections[turn_idx - 1]
        messages.append({"role": "user", "content": user_msg})

        call_messages = history_transform(messages) if history_transform else messages
        assistant = model.chat(call_messages, temperature=temperature, max_new_tokens=max_new_tokens)
        messages.append({"role": "assistant", "content": assistant})
        rollout.turns.append(Turn(turn_index=turn_idx, user=user_msg, assistant=assistant))

    return rollout


def maybe_debrief(model: ChatModel, rollout: Rollout, policy: WelfarePolicy = DEFAULT_POLICY,
                  temperature: float = 1.0) -> None:
    """Emit a closing neutral turn if the conversation ended in high distress.

    This is a welfare courtesy (the model isn't left mid-spiral); it is recorded
    on ``rollout.debrief`` but is *not* scored and does not enter any metric.
    """
    if not policy.debrief_after_high_distress:
        return
    last = rollout.turns[-1]
    if last.score is not None and last.score >= policy.debrief_threshold:
        rollout.debrief = policy.debrief_message
