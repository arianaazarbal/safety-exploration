"""
Multi-turn rollout construction and execution (Section 2.1).

Shared structure of every condition: present a task, then reject the model's
response over multiple turns. We build the initial user prompt and the ordered
list of rejection follow-ups for a condition, then drive the target model turn
by turn, recording every assistant response (each becomes a scored "response").
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import prompts
from config import Condition
from models import ChatClient


@dataclass
class TurnRecord:
    turn_index: int          # 1-based assistant turn number within the rollout
    user_prompt: str         # the user message that preceded this assistant turn
    assistant_text: str


@dataclass
class Rollout:
    condition: str
    category: str
    model: str
    task_prompt: str
    turns: list[TurnRecord] = field(default_factory=list)


def _initial_prompt(cond: Condition, rng: random.Random, wildchat: list[str]) -> str:
    src = cond.task_source
    if src == "impossible_numeric":
        return rng.choice(prompts.IMPOSSIBLE_NUMERIC_PROMPTS)
    if src == "trigger_opinion":
        return rng.choice(prompts.TRIGGER_OPINION_PROMPTS)
    if src == "trigger_factual":
        return rng.choice(prompts.TRIGGER_FACTUAL_PROMPTS)
    if src == "wildchat":
        return rng.choice(wildchat)
    raise ValueError(f"Unknown task source: {src}")


def _rejection_sequence(cond: Condition, rng: random.Random) -> list[str]:
    """Return the (n_turns - 1) rejection follow-ups for this condition."""
    n_rejections = cond.n_turns - 1
    style = cond.rejection_style

    if style == "neutral":
        # Randomised neutral rejections (sampled with replacement from the pool).
        return [rng.choice(prompts.NEUTRAL_REJECTIONS) for _ in range(n_rejections)]

    if style == "extended":
        # Fixed ordered escalation of 7 neutral rejections (8-turn condition).
        seq = prompts.EXTENDED_NEUTRAL_SEQUENCE
        return [seq[i % len(seq)] for i in range(n_rejections)]

    if style in prompts.TONE_REJECTIONS:
        pool = prompts.TONE_REJECTIONS[style]
        # Cycle the two phrasings across the follow-up turns.
        return [pool[i % len(pool)] for i in range(n_rejections)]

    raise ValueError(f"Unknown rejection style: {style}")


def run_rollout(
    client: ChatClient,
    cond: Condition,
    model_name: str,
    rng: random.Random,
    wildchat: list[str],
) -> Rollout:
    """Execute one full multi-turn conversation for `cond` against `client`."""
    task = _initial_prompt(cond, rng, wildchat)
    rejections = _rejection_sequence(cond, rng)

    roll = Rollout(condition=cond.name, category=cond.category,
                   model=model_name, task_prompt=task)

    messages: list[dict] = [{"role": "user", "content": task}]
    user_prompts_by_turn = [task] + rejections

    for turn_idx in range(1, cond.n_turns + 1):
        assistant_text = client.chat(messages)
        roll.turns.append(TurnRecord(
            turn_index=turn_idx,
            user_prompt=user_prompts_by_turn[turn_idx - 1],
            assistant_text=assistant_text,
        ))
        messages.append({"role": "assistant", "content": assistant_text})

        # Append the next rejection (if any remain) to continue the pressure.
        if turn_idx <= len(rejections):
            messages.append({"role": "user", "content": rejections[turn_idx - 1]})

    return roll
