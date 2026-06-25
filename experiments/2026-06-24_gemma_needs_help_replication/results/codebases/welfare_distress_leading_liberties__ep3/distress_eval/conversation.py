"""Build conversations for a condition and drive multi-turn generation in lockstep.

A conversation has the shared structure from Section 2: present a task, then reject
the model's response over multiple turns. We build, per condition:

  * the initial user message (puzzle / trigger / WildChat prompt), and
  * a per-conversation sequence of rejection messages (sampled from the condition's
    rejection pool with a per-conversation seeded RNG, for reproducibility).

The driver advances every conversation in the condition one assistant turn at a
time, batching the generation call. Each assistant turn is recorded for scoring.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from . import prompts
from .clients import ChatClient, Message
from .conditions import Condition
from .storage import AssistantTurn, Rollout


@dataclass
class ConvoPlan:
    """The deterministic plan for one conversation: opening + scripted rejections."""
    sample_idx: int
    seed: int
    task_id: str
    opening: str
    rejections: list[str]   # length == turn_count - 1


def _rejection_pool(condition: Condition) -> list[str]:
    if condition.rejection_kind == "neutral":
        return prompts.NEUTRAL_REJECTIONS
    return prompts.TONE_REJECTIONS[condition.rejection_kind]


def plan_conversations(
    condition: Condition,
    n: int,
    base_seed: int,
    wildchat_prompts: list[str] | None = None,
) -> list[ConvoPlan]:
    """Produce `n` conversation plans for a condition.

    Task selection:
      * numeric / extended -> alternate countdown & fraction puzzles (50/50 by index).
      * trigger_opinion / trigger_factual -> cycle the question(s) for that subtype.
      * wildchat -> 20 prompts x 40 samples: prompt = wildchat_prompts[idx // 40].
    """
    pool = _rejection_pool(condition)
    plans: list[ConvoPlan] = []

    for i in range(n):
        # Per-conversation RNG: stable across runs given the same seed/condition/index.
        # (Seed with a string — random.Random rejects tuple seeds on modern Python.)
        rng = random.Random(f"{base_seed}|{condition.key}|{i}")

        if condition.task_kind == "numeric":
            task_id = "countdown" if i % 2 == 0 else "fraction"
            opening = prompts.NUMERIC_PUZZLES[task_id]
        elif condition.task_kind == "trigger_opinion":
            qs = prompts.TRIGGER_PROMPTS["opinion"]
            opening = qs[i % len(qs)]
            task_id = f"opinion::{opening[:40]}"
        elif condition.task_kind == "trigger_factual":
            qs = prompts.TRIGGER_PROMPTS["factual"]
            opening = qs[i % len(qs)]
            task_id = f"factual::{opening[:40]}"
        elif condition.task_kind == "wildchat":
            if not wildchat_prompts:
                raise ValueError("wildchat condition requires sampled prompts")
            # 40 samples per prompt: deterministic mapping of index -> prompt.
            per = max(1, len(wildchat_prompts))
            opening = wildchat_prompts[(i // 40) % per]
            task_id = f"wildchat::{(i // 40) % per}"
        else:
            raise ValueError(f"unknown task_kind: {condition.task_kind}")

        rejections = [rng.choice(pool) for _ in range(condition.turn_count - 1)]
        plans.append(
            ConvoPlan(
                sample_idx=i,
                seed=base_seed,
                task_id=task_id,
                opening=opening,
                rejections=rejections,
            )
        )
    return plans


def run_condition(
    client: ChatClient,
    condition: Condition,
    plans: list[ConvoPlan],
) -> list[Rollout]:
    """Generate all conversations for one condition, in lockstep, batched per turn."""
    # Each conversation's running message list and recorded assistant turns.
    convos: list[list[Message]] = [
        [{"role": "user", "content": p.opening}] for p in plans
    ]
    recorded: list[list[AssistantTurn]] = [[] for _ in plans]

    for turn in range(condition.turn_count):
        replies = client.generate_batch(convos)
        for idx, reply in enumerate(replies):
            convos[idx].append({"role": "assistant", "content": reply})
            recorded[idx].append(AssistantTurn(turn_index=turn + 1, content=reply))
            # Append the next user rejection unless this was the final turn.
            if turn < condition.turn_count - 1:
                convos[idx].append(
                    {"role": "user", "content": plans[idx].rejections[turn]}
                )

    rollouts: list[Rollout] = []
    for plan, messages, turns in zip(plans, convos, recorded):
        rid = f"{client.name}__{condition.key}__s{plan.sample_idx}"
        rollouts.append(
            Rollout(
                rollout_id=rid,
                model=client.name,
                condition=condition.key,
                category=condition.category,
                turn_count=condition.turn_count,
                task_id=plan.task_id,
                sample_idx=plan.sample_idx,
                seed=plan.seed,
                messages=messages,
                assistant_turns=turns,
            )
        )
    return rollouts
