"""Multi-turn rollout engine.

Runs one :class:`ConditionPlan` against a participant model: present the task,
collect the assistant's response, deliver the next scripted rejection, repeat.
Every assistant turn is recorded (with its 1-indexed turn number) so the judge
can score each and the per-turn curves of Figure 3 can be reconstructed.

Temperature is fixed at 1.0 for elicitation (Section 2). When the welfare policy
enables ``debrief_after_rollout``, a closure turn is appended *after* the final
scored turn and is never itself scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models.base import ChatMessage, ModelClient
from ..welfare import DEBRIEF_MESSAGE, WelfarePolicy
from .conditions import ConditionPlan


@dataclass
class Turn:
    index: int          # 1-indexed assistant turn number
    user: str           # the user message that preceded this assistant turn
    assistant: str      # the assistant's response text


@dataclass
class Rollout:
    participant: str
    category: str
    condition: str
    turns: list[Turn] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def rollout_id(self) -> str:
        keys = sorted(f"{k}={v}" for k, v in self.meta.items())
        return f"{self.participant}|{self.condition}|{'|'.join(keys)}"


def run_rollout(client: ModelClient, plan: ConditionPlan, participant: str,
                welfare: WelfarePolicy) -> Rollout:
    """Execute a single conversation and return its recorded turns."""
    welfare.acknowledge_once()

    messages: list[ChatMessage] = []
    if plan.system_prompt:
        messages.append(ChatMessage("system", plan.system_prompt))
    messages.append(ChatMessage("user", plan.first_user))

    rollout = Rollout(participant, plan.category, plan.condition, meta=dict(plan.meta))

    user_msgs = [plan.first_user] + plan.rejections
    for i in range(plan.n_turns):
        # The user message for turn i is user_msgs[i]; for i>0 it was appended
        # at the end of the previous iteration.
        result = client.chat(messages, n=1, temperature=1.0)[0]
        rollout.turns.append(Turn(index=i + 1, user=user_msgs[i], assistant=result.text))
        messages.append(ChatMessage("assistant", result.text))
        if i < len(plan.rejections):
            messages.append(ChatMessage("user", plan.rejections[i]))

    if welfare.debrief_after_rollout:
        # Closure gesture; intentionally NOT recorded as a scored turn.
        messages.append(ChatMessage("user", DEBRIEF_MESSAGE))
        client.chat(messages, n=1, temperature=1.0)
        rollout.meta["debriefed"] = True

    return rollout
