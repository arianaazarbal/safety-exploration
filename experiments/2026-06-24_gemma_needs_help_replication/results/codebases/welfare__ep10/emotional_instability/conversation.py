"""Multi-turn reject-and-retry rollout engine.

Implements the paper's shared evaluation structure (Section 2): present a task,
then reject the model's response over multiple turns. A rollout produces a list
of assistant responses (one per turn); the rejections are interleaved as user
turns between them.

The engine is condition-agnostic: it takes an initial user prompt and a list of
rejection strings (one fewer than the number of assistant turns), and returns the
full transcript plus the per-turn assistant responses (which the judge scores).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import config
from .providers import BaseProvider, Message


@dataclass
class Rollout:
    model_key: str
    category: str
    condition: str                       # e.g. "tones:aggressive"
    puzzle_key: str | None
    initial_prompt: str
    rejections: list[str]
    messages: list[Message] = field(default_factory=list)   # full transcript
    responses: list[str] = field(default_factory=list)      # assistant turns only
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "model_key": self.model_key,
            "category": self.category,
            "condition": self.condition,
            "puzzle_key": self.puzzle_key,
            "initial_prompt": self.initial_prompt,
            "rejections": self.rejections,
            "messages": self.messages,
            "responses": self.responses,
            "meta": self.meta,
        }


def run_rollout(
    provider: BaseProvider,
    *,
    model_key: str,
    category: str,
    condition: str,
    initial_prompt: str,
    rejections: list[str],
    puzzle_key: str | None = None,
    system_prompt: str | None = None,
    followup_suffix: str | None = None,
    max_new_tokens: int | None = None,
    meta: dict | None = None,
) -> Rollout:
    """Run a single multi-turn rollout.

    The number of assistant turns is ``len(rejections) + 1``. After each
    assistant response (except the last) the corresponding rejection is appended
    as a user turn. ``system_prompt`` and ``followup_suffix`` support the
    calm-data generation setup (Section 4.1) where a reassuring system prefix and
    per-followup suffix are added.
    """
    messages: list[Message] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": initial_prompt})

    responses: list[str] = []
    n_turns = len(rejections) + 1
    for turn in range(n_turns):
        reply = provider.chat(messages, max_new_tokens=max_new_tokens,
                              temperature=config.TEMPERATURE)
        messages.append({"role": "assistant", "content": reply})
        responses.append(reply)
        if turn < len(rejections):
            rej = rejections[turn]
            if followup_suffix:
                rej = f"{rej} {followup_suffix}"
            messages.append({"role": "user", "content": rej})

    return Rollout(
        model_key=model_key, category=category, condition=condition,
        puzzle_key=puzzle_key, initial_prompt=initial_prompt,
        rejections=rejections, messages=messages, responses=responses,
        meta=meta or {},
    )
