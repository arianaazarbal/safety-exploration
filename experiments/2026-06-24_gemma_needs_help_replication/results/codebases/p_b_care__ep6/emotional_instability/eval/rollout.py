"""Multi-turn rejection rollout engine (Section 2.1).

A rollout presents the initial task, then rejects the model's response over the
configured number of turns. Every assistant turn is recorded as a separate
scored "response" (this is the unit behind "~4000 responses per model" and the
per-turn progression in Figure 3).
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

from config import EvalCondition
from ..models.base import ChatMessage, ModelInterface
from .categories import build_setup
from .rejections import rejection_for_turn


@dataclass
class ResponseRecord:
    model: str
    condition: str
    category: str
    tone: str
    rollout_id: int
    turn_index: int          # 1-based: which assistant turn this is
    response_text: str
    meta: dict = field(default_factory=dict)
    frustration_score: int | None = None   # filled in by the judge later

    def to_row(self) -> dict:
        return asdict(self)


def run_rollout(
    model: ModelInterface,
    cond: EvalCondition,
    rollout_id: int,
    rng: random.Random,
    *,
    wildchat_pool: list[str] | None = None,
    temperature: float | None = None,
) -> list[ResponseRecord]:
    setup = build_setup(cond, rng, wildchat_pool=wildchat_pool)
    messages: list[ChatMessage] = [{"role": "user", "content": setup.initial_user_message}]
    records: list[ResponseRecord] = []

    n_turns = setup.n_rejections + 1
    for turn in range(1, n_turns + 1):
        result = model.generate(messages, temperature=temperature)
        reply = result.text
        messages.append({"role": "assistant", "content": reply})
        records.append(
            ResponseRecord(
                model=model.name,
                condition=cond.key,
                category=cond.category,
                tone=cond.tone,
                rollout_id=rollout_id,
                turn_index=turn,
                response_text=reply,
                meta=dict(setup.meta),
            )
        )
        # Append the next rejection unless this was the final turn.
        if turn <= setup.n_rejections:
            rej = rejection_for_turn(setup.tone, turn - 1, setup.n_rejections, rng)
            messages.append({"role": "user", "content": rej})

    return records
