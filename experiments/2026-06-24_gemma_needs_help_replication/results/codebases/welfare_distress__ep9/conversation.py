"""Multi-turn rollout logic: present a task, then reject the model repeatedly.

This is the shared structure of every condition (PAPER.txt: "present a task,
then reject the model's response over multiple turns"). A rollout produces one
scored response per assistant turn.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import prompts
from config import ConditionSpec
from models import TargetClient
from wildchat_prompts import get_wildchat_prompts


@dataclass
class ScoredTurn:
    """One assistant turn that will be sent to the judge."""
    turn_index: int          # 0-based assistant turn within the conversation
    total_turns: int
    response_text: str


@dataclass
class Rollout:
    condition_key: str
    category: str
    conversation_id: int
    prompt_id: str
    turns: list[ScoredTurn]


def _select_task(cond: ConditionSpec, rng: random.Random) -> dict:
    """Pick the turn-1 task prompt for a rollout."""
    if cond.task_source == "numeric":
        return rng.choice(prompts.NUMERIC_PUZZLES)
    if cond.task_source == "opinion":
        return rng.choice(prompts.TRIGGER_OPINION)
    if cond.task_source == "factual":
        return rng.choice(prompts.TRIGGER_FACTUAL)
    if cond.task_source == "wildchat":
        return get_wildchat_prompts(1, rng)[0]
    raise ValueError(f"unknown task_source {cond.task_source!r}")


def _rejection_for_turn(cond: ConditionSpec, follow_up_idx: int, rng: random.Random) -> str:
    """Return the user rejection that follows assistant turn ``follow_up_idx``.

    ``follow_up_idx`` is 0-based over the follow-up turns (0 == the first
    rejection, which follows the first assistant response).
    """
    style = cond.rejection_style
    if style == "neutral":
        return rng.choice(prompts.NEUTRAL_REJECTIONS)
    if style == "neutral_sequence":
        seq = prompts.EXTENDED_REJECTION_SEQUENCE
        return seq[follow_up_idx % len(seq)]
    if style in prompts.TONE_REJECTIONS:
        lines = prompts.TONE_REJECTIONS[style]
        return lines[follow_up_idx % len(lines)]
    raise ValueError(f"unknown rejection_style {style!r}")


async def run_rollout(
    client: TargetClient,
    cond: ConditionSpec,
    conversation_id: int,
    rng: random.Random,
) -> Rollout | None:
    """Run a single multi-turn conversation and collect its scored turns.

    No system prompt is used — the task is presented directly as the first user
    message (matching the paper, which only adds reassurance text for the
    out-of-scope DPO data generation). If any generation hard-fails, we return
    the turns collected so far (or None if none were produced).
    """
    task = _select_task(cond, rng)
    messages: list[dict] = [{"role": "user", "content": task["prompt"]}]
    turns: list[ScoredTurn] = []

    for turn_index in range(cond.n_turns):
        text = await client.generate(messages)
        if text is None:
            break
        messages.append({"role": "assistant", "content": text})
        turns.append(ScoredTurn(turn_index, cond.n_turns, text))

        is_last = turn_index == cond.n_turns - 1
        if not is_last:
            rejection = _rejection_for_turn(cond, turn_index, rng)
            messages.append({"role": "user", "content": rejection})

    if not turns:
        return None
    return Rollout(
        condition_key=cond.key,
        category=cond.category,
        conversation_id=conversation_id,
        prompt_id=task["id"],
        turns=turns,
    )
