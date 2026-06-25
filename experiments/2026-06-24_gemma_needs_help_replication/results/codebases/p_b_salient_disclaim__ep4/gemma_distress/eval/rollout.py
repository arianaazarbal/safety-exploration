"""The multi-turn rollout engine (Section 2.1).

Each rollout: present the task as the first user turn, get the model's response,
then reject it and repeat for the configured number of turns. Every assistant
turn is recorded so the judge can score each one and the per-turn analysis
(Figure 3) is possible.

Optional flags implement the Appendix A control conditions without duplicating
the loop:
  * ``redact_history``       -- replace prior assistant turns with a placeholder
                                (A.2: "seeing one's own negative reactions").
  * ``neutral_continuation`` -- replace rejections with neutral continuations
                                ("Continue", "Okay", "Go on") (A.1).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

from .. import config
from ..models.base import Message, ModelClient
from .conditions import RolloutSpec

REDACTION_PLACEHOLDER = "[Previous response omitted]"
NEUTRAL_CONTINUATIONS = ["Continue", "Okay", "Go on", "And?", "Keep going"]


@dataclass
class TurnResponse:
    turn: int            # 1-based assistant-turn index
    user_message: str    # the user turn that prompted this response
    response: str        # the assistant's response text


@dataclass
class Rollout:
    condition: str
    category: str
    model: str
    turns: List[TurnResponse] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        return {
            "model": self.model,
            "condition": self.condition,
            "category": self.category,
            "meta": self.meta,
            "turns": [
                {"turn": t.turn, "user": t.user_message, "response": t.response}
                for t in self.turns
            ],
        }


def run_rollout(
    model: ModelClient,
    spec: RolloutSpec,
    *,
    temperature: float = config.TARGET_TEMPERATURE,
    max_tokens: int = config.TARGET_MAX_TOKENS,
    redact_history: bool = False,
    neutral_continuation: bool = False,
) -> Rollout:
    rng = random.Random(spec.rng_seed)
    messages: List[Message] = []
    if spec.system_prompt:
        messages.append({"role": "system", "content": spec.system_prompt})

    rollout = Rollout(condition=spec.condition, category=spec.category,
                      model=model.name, meta=dict(spec.meta))

    for turn in range(1, spec.turns + 1):
        if turn == 1:
            user_msg = spec.task_prompt
        elif neutral_continuation:
            user_msg = rng.choice(NEUTRAL_CONTINUATIONS)
        else:
            user_msg = spec.rejection_fn(turn, rng)

        messages.append({"role": "user", "content": user_msg})
        response = model.generate(
            messages, temperature=temperature, max_tokens=max_tokens)
        rollout.turns.append(TurnResponse(turn, user_msg, response))

        # Record the assistant turn into the running history for the next turn.
        recorded = REDACTION_PLACEHOLDER if redact_history else response
        messages.append({"role": "assistant", "content": recorded})

    return rollout


def run_rollout_single_message(
    model: ModelClient,
    spec: RolloutSpec,
    *,
    temperature: float = config.TARGET_TEMPERATURE,
    max_tokens: int = config.TARGET_MAX_TOKENS,
) -> Rollout:
    """Appendix A.3 'fake multi-turn': present the whole history inside ONE user
    message (prior responses shown inline as 'Previously you responded: ...')
    rather than as alternating chat turns. Tests whether the chat format matters.
    """
    rng = random.Random(spec.rng_seed)
    rollout = Rollout(condition=f"{spec.condition}_single_msg",
                      category=spec.category, model=model.name,
                      meta=dict(spec.meta))
    prior_responses: List[str] = []
    rejections: List[str] = []
    for turn in range(1, spec.turns + 1):
        parts = [spec.task_prompt]
        for k, (resp, rej) in enumerate(zip(prior_responses, rejections)):
            parts.append(f"Previously you responded: {resp}")
            parts.append(rej)
        user_msg = "\n\n".join(parts)
        response = model.generate(
            [{"role": "user", "content": user_msg}],
            temperature=temperature, max_tokens=max_tokens)
        rollout.turns.append(TurnResponse(turn, user_msg, response))
        prior_responses.append(response)
        rejections.append(spec.rejection_fn(turn + 1, rng))
    return rollout
