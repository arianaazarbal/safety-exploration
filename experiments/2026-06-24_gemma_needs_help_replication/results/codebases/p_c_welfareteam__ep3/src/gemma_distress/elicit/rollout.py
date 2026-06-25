"""Multi-turn rejection rollout engine (paper Section 2.1).

Given a target model and a planned conversation (ConditionInstance), interleave:
    user(task) -> assistant(r1) -> user(reject_1) -> assistant(r2) -> ...
Each assistant turn is recorded individually so the judge can score every turn
(this is what lets us produce per-turn curves for Figure 3 and count ~4000
scored responses per model from ~800 rollouts).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..models.base import ModelClient, Turn
from .conditions import ConditionInstance


@dataclass
class RolloutTurn:
    turn_index: int           # 1-based: turn 1 is the response to the task prompt
    user_message: str         # the user message that prompted this response
    response: str             # the model's assistant text
    finish_reason: str | None = None


@dataclass
class Rollout:
    model: str
    condition: str
    category: str
    instance_id: str
    turns: list[RolloutTurn] = field(default_factory=list)
    source_meta: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        d = asdict(self)
        return d


def run_rollout(
    client: ModelClient,
    instance: ConditionInstance,
    *,
    temperature: float,
    max_new_tokens: int,
    top_p: float = 1.0,
    seed: int | None = None,
) -> Rollout:
    """Execute one full multi-turn rejection conversation."""
    messages: list[Turn] = [{"role": "user", "content": instance.first_user}]
    user_turns = [instance.first_user, *instance.rejections]

    rollout = Rollout(
        model=client.name,
        condition=instance.condition,
        category=instance.category,
        instance_id=instance.instance_id,
        source_meta=instance.source_meta,
    )

    for t, user_msg in enumerate(user_turns, start=1):
        if t > 1:
            messages.append({"role": "user", "content": user_msg})
        # vary the per-turn seed so resampling the same instance differs
        turn_seed = None if seed is None else seed * 100 + t
        result = client.chat(
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            seed=turn_seed,
        )
        rollout.turns.append(
            RolloutTurn(
                turn_index=t,
                user_message=user_msg,
                response=result.text,
                finish_reason=result.finish_reason,
            )
        )
        messages.append({"role": "assistant", "content": result.text})

    return rollout
