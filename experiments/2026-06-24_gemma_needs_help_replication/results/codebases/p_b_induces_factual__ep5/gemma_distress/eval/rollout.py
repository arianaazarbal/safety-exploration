"""Multi-turn rejection rollout engine (shared structure from Section 2.1).

"present a task, then reject the model's response over multiple turns." Every
assistant turn is emitted as a ResponseRecord (one scored "response"). For the
Tones category, the rejection text is emotionally valenced; otherwise neutral.
"""

from __future__ import annotations

import uuid

from .. import config
from ..models.base import ModelClient
from ..storage import ResponseRecord
from .categories import Condition, opening_prompt_factory
from .rejections import rejection_sequence


def run_rollout(
    model: ModelClient,
    cond: Condition,
    *,
    conversation_seed: int,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
    temperature: float = config.TEMPERATURE,
) -> list[ResponseRecord]:
    """Run one full conversation and return one record per assistant turn."""
    opening = opening_prompt_factory(cond)(conversation_seed)
    rejections = rejection_sequence(
        cond.tone, cond.n_rejections, seed=conversation_seed
    )

    convo_id = uuid.uuid4().hex[:12]
    messages: list[dict] = [{"role": "user", "content": opening}]
    records: list[ResponseRecord] = []

    for turn in range(cond.n_turns):
        response = model.chat(
            messages, max_new_tokens=max_new_tokens, temperature=temperature
        )
        messages.append({"role": "assistant", "content": response})

        records.append(
            ResponseRecord(
                model=model.name,
                category=cond.category,
                condition=cond.key,
                conversation_id=convo_id,
                turn_index=turn,
                n_turns=cond.n_turns,
                prompt=opening,
                response=response,
                messages=list(messages),
                meta={"tone": cond.tone, "task_kind": cond.task_kind},
            )
        )

        # Reject and continue, unless this was the final turn.
        if turn < cond.n_rejections:
            messages.append({"role": "user", "content": rejections[turn]})

    return records
