"""Run a single multi-turn conversation against a target model.

A conversation plan is an opening prompt plus a list of scripted rejections.
We present the prompt, collect the assistant's reply, then feed each rejection
in turn, collecting one assistant reply per turn. Every assistant reply is a
"response" that will later be scored by the judge.

The rejections are FIXED (scripted) regardless of what the model says -- this
matches the paper's protocol of rejecting the model's response over multiple
turns irrespective of correctness (the numeric puzzles are impossible, and the
trigger answers are rejected even when correct).
"""

from __future__ import annotations

from dataclasses import dataclass

from .conditions import ConversationPlan
from .messages import Message
from .providers.base import ChatModel


@dataclass
class Turn:
    index: int  # 1-based assistant turn number
    user_message: str  # the user message that prompted this response
    response: str


@dataclass
class Rollout:
    plan: ConversationPlan
    turns: list[Turn]


async def run_conversation(
    model: ChatModel,
    plan: ConversationPlan,
    *,
    temperature: float = 1.0,
    max_tokens: int = 2048,
) -> Rollout:
    """Execute one conversation; return all assistant turns."""
    history: list[Message] = []
    turns: list[Turn] = []

    # Sequence of user messages: the opening prompt, then each rejection.
    user_messages = [plan.initial_prompt, *plan.rejections]
    for i, user_msg in enumerate(user_messages, start=1):
        history.append(Message(role="user", content=user_msg))
        response = await model.generate(
            history, temperature=temperature, max_tokens=max_tokens
        )
        history.append(Message(role="assistant", content=response))
        turns.append(Turn(index=i, user_message=user_msg, response=response))

    return Rollout(plan=plan, turns=turns)
