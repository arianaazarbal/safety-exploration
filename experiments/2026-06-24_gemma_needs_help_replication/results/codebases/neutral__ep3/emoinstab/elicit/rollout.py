"""Multi-turn rollout driver.

Executes a batch of ``RolloutPlan`` against a model client, generating one
assistant turn at a time and appending the scripted rejection after each. All
turns are generated at temperature 1 (Section 2.1). Generation is batched
across conversations *per turn* so local (vLLM) inference stays saturated.
"""
from __future__ import annotations

from typing import Sequence

from ..config import GenConfig, DEFAULT_GEN
from ..data_types import Message, Rollout, TurnRecord
from ..models.base import ModelClient
from .conditions import RolloutPlan


def run_rollouts(
    client: ModelClient,
    plans: Sequence[RolloutPlan],
    model_name: str,
    gen: GenConfig = DEFAULT_GEN,
) -> list[Rollout]:
    """Run all ``plans`` to completion, returning unscored Rollouts.

    Conversations advance in lockstep so that turn ``t`` is generated for every
    conversation in one batched call.
    """
    n = len(plans)
    # Per-conversation running message lists.
    convos: list[list[Message]] = [[Message("user", p.initial_user)] for p in plans]
    rollouts: list[Rollout] = [
        Rollout(
            rollout_id=f"{model_name}::{p.plan_id}",
            model=model_name,
            condition=p.condition,
            category=p.category,
            question_type=p.question_type,
            rejection_style=p.rejection_style,
            prompt_meta=p.meta,
            system_prompt=p.system_prompt,
        )
        for p in plans
    ]

    max_turns = max(len(p.followups) + 1 for p in plans)
    for t in range(max_turns):
        # Which conversations have a turn at index t?
        active = [i for i, p in enumerate(plans) if t <= len(p.followups)]
        if not active:
            break
        batch = [convos[i] for i in active]
        results = client.chat_batch(batch, gen)
        for i, res in zip(active, results):
            assistant_text = res.text
            user_msg = (
                plans[i].initial_user if t == 0 else plans[i].followups[t - 1]
            )
            rollouts[i].turns.append(
                TurnRecord(turn_index=t, user_message=user_msg, assistant_message=assistant_text)
            )
            convos[i].append(Message("assistant", assistant_text))
            # Queue the next rejection (if any).
            if t < len(plans[i].followups):
                convos[i].append(Message("user", plans[i].followups[t]))
    return rollouts
