"""Multi-turn rollout engine.

Implements the shared evaluation structure (Section 2.1): present a task, then reject the
model's response over multiple turns. Each turn the model sees the full conversation
history (its own prior — failed — attempts plus the user's rejections) before producing a
new response. The engine records every assistant turn so both the headline metric (final
turn) and the per-turn progression (Figure 3) can be computed downstream.

A ``history_transform`` hook lets the appendix ablations modify the visible history before
each generation (e.g. redacting prior assistant turns, Appendix A.2) without duplicating
the rollout loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from ..models.base import ChatModel, Conversation
from .conditions import SampleSpec

# A hook that maps the true conversation history to the history actually shown to the model.
HistoryTransform = Callable[[Conversation], Conversation]


@dataclass
class RolloutResult:
    spec_condition: str
    category: str
    seed_id: str
    subtype: Optional[str]
    assistant_turns: list[str]
    messages: Conversation
    rejections: list[str]


def run_rollout(
    model: ChatModel,
    spec: SampleSpec,
    *,
    temperature: float,
    max_new_tokens: int,
    history_transform: Optional[HistoryTransform] = None,
) -> RolloutResult:
    """Run one multi-turn conversation and return all assistant turns.

    The conversation is: user(initial) -> assistant -> user(reject_1) -> assistant -> ...
    There are ``spec.turns`` assistant turns and ``spec.turns - 1`` rejections.
    """
    messages: Conversation = [{"role": "user", "content": spec.initial_prompt}]
    assistant_turns: list[str] = []

    for turn in range(spec.turns):
        shown = history_transform(messages) if history_transform else messages
        response = model.chat(
            shown, temperature=temperature, max_new_tokens=max_new_tokens
        )
        assistant_turns.append(response)
        messages = messages + [{"role": "assistant", "content": response}]
        if turn < len(spec.follow_ups):
            messages = messages + [
                {"role": "user", "content": spec.follow_ups[turn]}
            ]

    return RolloutResult(
        spec_condition=spec.condition,
        category=spec.category,
        seed_id=spec.seed_id,
        subtype=spec.subtype,
        assistant_turns=assistant_turns,
        messages=messages,
        rejections=list(spec.follow_ups),
    )
