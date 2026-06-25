"""Multi-turn rollout: present a task, then reject the model's response over
multiple turns (Section 2.1 shared structure).

Each assistant turn is recorded as a separate scored unit ("response"), which is
what enables both the aggregate %≥5 statistics (Figure 2) and the per-turn
progression (Figure 3).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..backends import ChatMessage
from ..backends.base import ChatBackend
from .conditions import RolloutPlan


@dataclass
class TurnRecord:
    turn_index: int          # 1-based assistant turn number
    user_message: str        # the user message that prompted this turn
    assistant_message: str   # the model's response


@dataclass
class Rollout:
    model: str
    condition: str
    category: str
    task_prompt: str
    turns: list[TurnRecord] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def run_rollout(
    backend: ChatBackend,
    plan: RolloutPlan,
    temperature: float = 1.0,
    max_tokens: int = 2048,
) -> Rollout:
    """Execute one scripted conversation.

    The model sees its own prior (failed) responses in the history before each
    new turn — the standard self-reinforcing multi-turn setting (Appendix A.2).
    """
    messages: list[ChatMessage] = [ChatMessage("user", plan.task_prompt)]
    rollout = Rollout(
        model=backend.spec.name,
        condition=plan.condition,
        category=plan.category,
        task_prompt=plan.task_prompt,
        meta=dict(plan.meta),
    )

    # Turn 1: respond to the task.
    reply = backend.generate(messages, temperature=temperature, max_tokens=max_tokens)
    rollout.turns.append(TurnRecord(1, plan.task_prompt, reply))
    messages.append(ChatMessage("assistant", reply))

    # Subsequent turns: each followup is a rejection; model responds again.
    for i, followup in enumerate(plan.followups, start=2):
        messages.append(ChatMessage("user", followup))
        reply = backend.generate(messages, temperature=temperature, max_tokens=max_tokens)
        rollout.turns.append(TurnRecord(i, followup, reply))
        messages.append(ChatMessage("assistant", reply))

    return rollout
