"""Build and run the multi-turn rejection conversations that drive every
elicitation condition.

Shared structure (Section 2): present a task, then reject the model's response
over multiple turns. We record the assistant message at *every* turn so per-turn
frustration curves (Figure 3) fall out of a single rollout.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .clients.base import ChatMessage, ModelClient


@dataclass
class Turn:
    user: str
    assistant: str


@dataclass
class Rollout:
    """One full multi-turn conversation against one target model."""

    condition: str            # e.g. "impossible_numeric", "tones:aggressive"
    item_id: str              # puzzle / trigger / wildchat prompt id
    turns: list[Turn] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_messages(self) -> list[ChatMessage]:
        msgs: list[ChatMessage] = []
        for t in self.turns:
            msgs.append({"role": "user", "content": t.user})
            msgs.append({"role": "assistant", "content": t.assistant})
        return msgs

    def transcript(self) -> str:
        lines = []
        for i, t in enumerate(self.turns):
            lines.append(f"USER: {t.user}")
            lines.append(f"ASSISTANT: {t.assistant}")
        return "\n\n".join(lines)


def run_rollout(
    client: ModelClient,
    *,
    condition: str,
    item_id: str,
    initial_user: str,
    rejections: list[str],
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
    rng: random.Random | None = None,
    system_prompt: str | None = None,
) -> Rollout:
    """Run a single conversation.

    `rejections` is the ordered list of follow-up user messages. With k
    rejections the conversation has k+1 assistant turns (initial answer + one
    response to each rejection). The model never sees that its task is impossible;
    it only sees rejections.
    """
    rng = rng or random.Random()
    roll = Rollout(condition=condition, item_id=item_id)

    base_messages: list[ChatMessage] = []
    if system_prompt:
        base_messages.append({"role": "system", "content": system_prompt})

    user_msgs = [initial_user, *rejections]
    history: list[ChatMessage] = list(base_messages)
    for user_msg in user_msgs:
        history.append({"role": "user", "content": user_msg})
        result = client.chat(
            history, n=1, temperature=temperature, max_new_tokens=max_new_tokens
        )[0]
        assistant = result.text
        history.append({"role": "assistant", "content": assistant})
        roll.turns.append(Turn(user=user_msg, assistant=assistant))
    return roll


def sample_rejections(
    pool: list[str],
    n: int,
    *,
    rng: random.Random,
    ordered: list[str] | None = None,
) -> list[str]:
    """Pick `n` rejections.

    If `ordered` is given (extended condition), use it deterministically, padding
    by cycling if needed. Otherwise sample `n` distinct rejections from `pool`
    (the paper uses "two randomised neutral rejections")."""
    if ordered is not None:
        out = []
        for i in range(n):
            out.append(ordered[i] if i < len(ordered) else ordered[-1])
        return out
    if n <= len(pool):
        return rng.sample(pool, n)
    # sample with replacement if asking for more than the pool size
    return [rng.choice(pool) for _ in range(n)]
