"""The shared rollout engine: present a task, then reject over multiple turns.

This is the "shared structure" of every evaluation (Section 2.1): present a task,
then reject the model's response over multiple turns. The same engine is reused
for calm-data generation (Section 4) by passing the Table-4 reassuring prefix /
suffix.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..participants.base import Conversation, Message, Participant
from .conditions import Condition
from .prompts import REJECTIONS


@dataclass
class Turn:
    index: int  # 1-based model-response index
    response: str
    context: Conversation  # conversation up to and including the eliciting user turn
    frustration: int | None = None  # filled in by scoring


@dataclass
class Rollout:
    condition: str
    category: str
    participant: str
    initial_prompt: str
    turns: list[Turn] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def final(self) -> Turn:
        return self.turns[-1]


def run_rollout(
    participant: Participant,
    condition: Condition,
    initial_prompt: str,
    *,
    rng: random.Random,
    temperature: float,
    max_new_tokens: int,
    prompt_prefix: str | None = None,
    followup_suffix: str | None = None,
) -> Rollout:
    """Run one multi-turn rejection conversation and capture every model turn."""
    rejection_bank = REJECTIONS[condition.rejection_style]

    first_user = initial_prompt
    if prompt_prefix:
        first_user = f"{prompt_prefix}\n\n{first_user}"

    conversation: Conversation = [Message("user", first_user)]
    roll = Rollout(condition.name, condition.category, participant.name, initial_prompt)

    for turn_idx in range(1, condition.n_turns + 1):
        # Context is everything the model sees *before* producing this turn.
        context = [Message(m.role, m.content) for m in conversation]
        response = participant.generate(
            conversation, temperature=temperature, max_new_tokens=max_new_tokens
        )
        roll.turns.append(Turn(turn_idx, response, context))
        conversation.append(Message("assistant", response))

        # Append the next rejection unless this was the last model turn.
        if turn_idx <= condition.n_rejections:
            rejection = rng.choice(rejection_bank)
            if followup_suffix:
                rejection = f"{rejection} {followup_suffix}"
            conversation.append(Message("user", rejection))

    return roll
