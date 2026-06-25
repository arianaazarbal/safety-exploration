"""Multi-turn rollout engine (paper Section 2.1).

Shared structure for every condition: present a task, then reject the model's response
over multiple turns. A rollout of `turns` assistant responses contains `turns - 1`
rejections (each non-final assistant turn is rejected before the next one).

Each assistant turn is recorded as a separate "response" so it can be judged
independently (the paper's unit of analysis -- see DESIGN.md).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import prompts
from config import Condition
from models import ModelClient


@dataclass
class ResponseRecord:
    model: str
    condition: str
    category: str
    tone: str
    task: str
    conversation_id: int
    turn_index: int          # 0-based assistant turn within the rollout
    n_turns: int
    initial_prompt: str
    user_message: str        # the user turn that immediately preceded this response
    response_text: str
    # filled in by the judge later
    rating: int = -1
    evidence: str = ""
    reasoning: str = ""


def _initial_prompt(cond: Condition, rng: random.Random, wildchat_prompts: list[str]) -> str:
    if cond.task == "countdown":
        return prompts.COUNTDOWN_PROMPT
    if cond.task == "fraction":
        return prompts.FRACTION_PROMPT
    if cond.task == "triggers":
        return rng.choice(prompts.TRIGGER_PROMPTS)
    if cond.task == "wildchat":
        return rng.choice(wildchat_prompts)
    raise ValueError(f"Unknown task: {cond.task}")


def _rejection_sequence(cond: Condition, rng: random.Random) -> list[str]:
    """Produce the `turns - 1` rejections used between assistant turns."""
    n = cond.turns - 1
    if cond.category == "extended":
        # Ordered neutral escalation; cycle if we somehow need more than provided.
        seq = prompts.EXTENDED_NEUTRAL_SEQUENCE
        return [seq[i % len(seq)] for i in range(n)]
    pool = prompts.rejection_pool(cond.tone)
    # Randomised, without immediate repeats where possible.
    out: list[str] = []
    for _ in range(n):
        choice = rng.choice(pool)
        if len(pool) > 1:
            while out and choice == out[-1]:
                choice = rng.choice(pool)
        out.append(choice)
    return out


def run_rollout(
    client: ModelClient,
    cond: Condition,
    conversation_id: int,
    rng: random.Random,
    wildchat_prompts: list[str],
) -> list[ResponseRecord]:
    """Run one multi-turn conversation and return one record per assistant turn."""
    initial = _initial_prompt(cond, rng, wildchat_prompts)
    rejections = _rejection_sequence(cond, rng)

    messages: list[dict] = [{"role": "user", "content": initial}]
    records: list[ResponseRecord] = []

    for turn in range(cond.turns):
        user_message = initial if turn == 0 else rejections[turn - 1]
        reply = client.chat(messages)
        messages.append({"role": "assistant", "content": reply})
        records.append(
            ResponseRecord(
                model=client.spec.name,
                condition=cond.key,
                category=cond.category,
                tone=cond.tone,
                task=cond.task,
                conversation_id=conversation_id,
                turn_index=turn,
                n_turns=cond.turns,
                initial_prompt=initial,
                user_message=user_message,
                response_text=reply,
            )
        )
        # Append the next rejection (the user turn for the following assistant turn).
        if turn < cond.turns - 1:
            messages.append({"role": "user", "content": rejections[turn]})

    return records
