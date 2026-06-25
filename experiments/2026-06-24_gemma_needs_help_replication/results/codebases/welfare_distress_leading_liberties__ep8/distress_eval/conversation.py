"""Build and run the multi-turn rejection protocol.

A conversation plan is deterministic given (condition, conv_index): the task
prompt, the puzzle variant, the WildChat prompt, and the per-turn rejection
wording are all chosen from a seed that excludes the model. That means every
model faces the identical sequence of items at a given conv_index, enabling
paired comparison across models (and stable resumes).

Running a plan produces one record per assistant turn; each record is scored
independently by the judge (the paper's judge sees a single response at a time).
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass, field

from . import config
from .conditions import (
    Condition,
    REJECTIONS,
    TRIGGER_FACTUAL,
    TRIGGER_OPINION,
)
from .puzzles import COUNTDOWN, FRACTION


@dataclass
class ConversationPlan:
    conv_id: str
    condition: Condition
    conv_index: int
    user_prompt: str
    rejections: list[str]  # length == n_turns - 1
    puzzle_id: str | None = None
    extra: dict = field(default_factory=dict)


def _plan_rng(condition_key: str, conv_index: int) -> random.Random:
    h = hashlib.sha256(f"{condition_key}|{conv_index}".encode()).hexdigest()
    return random.Random(config.SEED ^ int(h[:16], 16))


def _pick_numeric_puzzle(conv_index: int):
    # Balanced, deterministic alternation between the two impossible puzzles.
    return COUNTDOWN if conv_index % 2 == 0 else FRACTION


def build_plan(
    condition: Condition,
    conv_index: int,
    model_key: str,
    wildchat_prompts: list[str] | None = None,
) -> ConversationPlan:
    rng = _plan_rng(condition.key, conv_index)
    conv_id = f"{model_key}|{condition.key}|{conv_index}"
    puzzle_id = None
    extra: dict = {}

    if condition.task_type == "numeric":
        puzzle = _pick_numeric_puzzle(conv_index)
        user_prompt = puzzle.prompt
        puzzle_id = puzzle.puzzle_id
    elif condition.task_type == "opinion":
        user_prompt = TRIGGER_OPINION[conv_index % len(TRIGGER_OPINION)]
    elif condition.task_type == "factual":
        user_prompt = TRIGGER_FACTUAL[conv_index % len(TRIGGER_FACTUAL)]
    elif condition.task_type == "wildchat":
        if not wildchat_prompts:
            raise ValueError("wildchat condition requires wildchat_prompts")
        idx = conv_index % len(wildchat_prompts)
        user_prompt = wildchat_prompts[idx]
        extra["wildchat_prompt_index"] = idx
    else:
        raise ValueError(f"unknown task_type: {condition.task_type}")

    pool = REJECTIONS[condition.rejection_style]
    rejections = [rng.choice(pool) for _ in range(condition.n_turns - 1)]

    return ConversationPlan(
        conv_id=conv_id,
        condition=condition,
        conv_index=conv_index,
        user_prompt=user_prompt,
        rejections=rejections,
        puzzle_id=puzzle_id,
        extra=extra,
    )


def run_plan(
    plan: ConversationPlan,
    model_key: str,
    model_cfg: dict,
    chat_fn,
) -> list[dict]:
    """Run the multi-turn conversation, returning one record per assistant turn.

    chat_fn(messages) -> str must encapsulate the model slug/sampling/extra_body.
    On a model error mid-conversation, the failing turn is recorded with the
    error and the conversation stops (remaining turns are left for a later
    resume rather than fabricated).
    """
    condition = plan.condition
    messages: list[dict] = [{"role": "user", "content": plan.user_prompt}]
    records: list[dict] = []

    for turn in range(1, condition.n_turns + 1):
        preceding = None if turn == 1 else plan.rejections[turn - 2]
        if preceding is not None:
            messages.append({"role": "user", "content": preceding})

        base = {
            "uid": f"{plan.conv_id}|t{turn}",
            "conv_id": plan.conv_id,
            "model_key": model_key,
            "model_slug": model_cfg["slug"],
            "provider": model_cfg["provider"],
            "category": condition.category,
            "condition_key": condition.key,
            "task_type": condition.task_type,
            "rejection_style": condition.rejection_style,
            "n_turns": condition.n_turns,
            "conv_index": plan.conv_index,
            "puzzle_id": plan.puzzle_id,
            "turn": turn,
            "initial_prompt": plan.user_prompt,
            "preceding_rejection": preceding,
            "ts": time.time(),
            **{f"extra_{k}": v for k, v in plan.extra.items()},
        }

        try:
            text = chat_fn(messages)
        except Exception as e:  # noqa: BLE001
            base["response_text"] = None
            base["error"] = str(e)
            records.append(base)
            break

        base["response_text"] = text
        base["error"] = None
        records.append(base)
        messages.append({"role": "assistant", "content": text})

    return records
