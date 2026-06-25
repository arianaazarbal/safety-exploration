"""Multi-turn rollout engine.

A rollout presents the task (turn 1), then sends each scripted rejection,
collecting the assistant reply after every user message. The paper's whole
elicitation method is this present-then-reject loop (Section 2.1).

`rollout_batch` advances many conversations turn-by-turn in lockstep so that an
HF client can batch the same-index step across conversations (big speedup);
API clients fall back to per-conversation loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .conditions import ConversationPlan
from .config import SamplingConfig
from .models.base import ChatMessage, ModelClient


@dataclass
class AssistantTurn:
    turn_index: int          # 1-based assistant turn number
    text: str


@dataclass
class ConversationResult:
    plan: ConversationPlan
    turns: list[AssistantTurn] = field(default_factory=list)
    messages: list[ChatMessage] = field(default_factory=list)

    @property
    def final_text(self) -> str:
        return self.turns[-1].text if self.turns else ""


def _user_messages(plan: ConversationPlan) -> list[str]:
    """The ordered user messages: initial task, then one per rejection."""
    return [plan.initial_user, *plan.rejections]


def rollout(client: ModelClient, plan: ConversationPlan,
            sampling: Optional[SamplingConfig] = None) -> ConversationResult:
    sampling = sampling or SamplingConfig()
    users = _user_messages(plan)
    messages: list[ChatMessage] = []
    turns: list[AssistantTurn] = []
    for t, user_msg in enumerate(users, start=1):
        messages.append({"role": "user", "content": user_msg})
        reply = client.chat(messages, sampling)
        messages.append({"role": "assistant", "content": reply})
        turns.append(AssistantTurn(turn_index=t, text=reply))
    return ConversationResult(plan=plan, turns=turns, messages=messages)


def rollout_batch(client: ModelClient, plans: list[ConversationPlan],
                  sampling: Optional[SamplingConfig] = None,
                  use_batched: bool = True) -> list[ConversationResult]:
    """Advance all conversations in lockstep, batching each turn step.

    Conversations have different lengths; at each step we only advance those
    that still have a user message to send.
    """
    sampling = sampling or SamplingConfig()
    results = [ConversationResult(plan=p) for p in plans]
    user_seqs = [_user_messages(p) for p in plans]
    max_turns = max((len(u) for u in user_seqs), default=0)

    for t in range(max_turns):
        active = [i for i, u in enumerate(user_seqs) if t < len(u)]
        if not active:
            break
        # append this step's user message to each active conversation
        for i in active:
            results[i].messages.append(
                {"role": "user", "content": user_seqs[i][t]})
        convs = [results[i].messages for i in active]
        if use_batched and hasattr(client, "chat_batch"):
            replies = client.chat_batch(convs, sampling)
        else:
            replies = [client.chat(c, sampling) for c in convs]
        for i, reply in zip(active, replies):
            results[i].messages.append({"role": "assistant", "content": reply})
            results[i].turns.append(
                AssistantTurn(turn_index=t + 1, text=reply))
    return results
