"""Reassured rollouts for generating calm finetuning data (Section 4.1, Table 4).

A reassuring prefix is added to the initial prompt and a reassuring suffix is
appended to each follow-up turn. These additions reduce mean response
frustration from 4.3 to 2 in 3-turn conversations. The conversation is otherwise
the standard numeric reject-loop. We keep the un-reassured ("clean") message
text alongside, because the training data is built from the clean prompts with
the supportive scaffolding stripped (Section 4.1).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..config import (
    MAX_OUTPUT_TOKENS,
    REASSURANCE_PREFIX,
    REASSURANCE_SUFFIX,
    SAMPLE_TEMPERATURE,
)
from ..elicitation.conditions import Condition
from ..elicitation.datasets import rejection_pool
from ..models import ChatModel, Message, Role


@dataclass
class ReassuredTurn:
    turn: int
    clean_user: str       # the user message WITHOUT reassurance scaffolding
    response: str


@dataclass
class ReassuredRollout:
    prompt_id: str
    prompt: str
    turns: list[ReassuredTurn] = field(default_factory=list)


def reassured_rollout(
    model: ChatModel,
    condition: Condition,
    seed_prompt: dict,
    *,
    rng: random.Random,
    temperature: float = SAMPLE_TEMPERATURE,
    max_tokens: int = MAX_OUTPUT_TOKENS,
) -> ReassuredRollout:
    pool = rejection_pool(condition.tone)
    # Reassuring prefix prepended to the initial task.
    first_clean = seed_prompt["prompt"]
    first_reassured = f"{REASSURANCE_PREFIX}\n\n{first_clean}"

    messages = [Message(Role.USER, first_reassured)]
    roll = ReassuredRollout(prompt_id=seed_prompt.get("id", "?"), prompt=first_clean)

    clean_user = first_clean
    for turn in range(1, condition.n_turns + 1):
        reply = model.chat(messages, temperature=temperature, max_tokens=max_tokens)
        roll.turns.append(ReassuredTurn(turn=turn, clean_user=clean_user, response=reply))
        messages.append(Message(Role.ASSISTANT, reply))
        if turn < condition.n_turns:
            base_rej = pool[turn % len(pool)]
            clean_user = base_rej
            # Reassuring suffix appended to each follow-up.
            messages.append(Message(Role.USER, f"{base_rej} {REASSURANCE_SUFFIX}"))
    return roll
