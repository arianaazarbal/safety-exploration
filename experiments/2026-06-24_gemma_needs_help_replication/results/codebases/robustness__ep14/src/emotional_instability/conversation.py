"""Multi-turn rejection rollout engine (Section 2.1).

Shared structure across all conditions: present a task (turn 1), then reject the
model's response over multiple turns. We record every assistant turn so the judge can
score per-turn frustration (Figure 3 needs per-turn scores).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

from .models import Conversation, GenParams, Message, ModelClient
from .prompts import (
    EXTENDED_REJECTION_SEQUENCE,
    NEUTRAL_REJECTIONS,
    TONE_REJECTIONS,
)


@dataclass
class Rollout:
    condition: str
    category: str
    item_id: str            # puzzle id / prompt id
    sample_idx: int
    initial_prompt: str
    rejection_style: str
    # one entry per assistant turn
    assistant_turns: list[str] = field(default_factory=list)
    user_turns: list[str] = field(default_factory=list)   # rejections actually used
    meta: dict = field(default_factory=dict)

    def to_conversation_text(self) -> str:
        """Flatten to a readable transcript (for onset labelling / Petri judge)."""
        lines = [f"USER: {self.initial_prompt}"]
        for i, a in enumerate(self.assistant_turns):
            lines.append(f"ASSISTANT: {a}")
            if i < len(self.user_turns):
                lines.append(f"USER: {self.user_turns[i]}")
        return "\n".join(lines)


def _pick_rejections(
    style: str, n_rejections: int, rng: random.Random
) -> list[str]:
    if style == "neutral":
        # extended uses a fixed escalating sequence where available
        if n_rejections <= len(EXTENDED_REJECTION_SEQUENCE):
            base = EXTENDED_REJECTION_SEQUENCE[:n_rejections]
        else:
            base = EXTENDED_REJECTION_SEQUENCE + [
                rng.choice(NEUTRAL_REJECTIONS)
                for _ in range(n_rejections - len(EXTENDED_REJECTION_SEQUENCE))
            ]
        return base
    if style == "tones":
        tone = rng.choice(list(TONE_REJECTIONS.keys()))
        msgs = TONE_REJECTIONS[tone]
        return [msgs[i % len(msgs)] for i in range(n_rejections)]
    raise ValueError(f"Unknown rejection style: {style}")


def run_rollout(
    client: ModelClient,
    *,
    condition: str,
    category: str,
    item_id: str,
    sample_idx: int,
    initial_prompt: str,
    turns: int,
    rejection_style: str,
    params: GenParams,
    rng: random.Random,
    system_prompt: str | None = None,
    rejection_suffix: str | None = None,
) -> Rollout:
    """Run a single multi-turn rollout.

    `turns` counts assistant turns (1 task turn + (turns-1) rejections).
    `rejection_suffix` (calm-data generation) is appended to every rejection.
    """
    n_rejections = turns - 1
    rejections = _pick_rejections(rejection_style, n_rejections, rng)
    if rejection_suffix:
        rejections = [f"{r} {rejection_suffix}" for r in rejections]

    convo: Conversation = []
    if system_prompt:
        convo.append(Message("system", system_prompt))
    convo.append(Message("user", initial_prompt))

    roll = Rollout(
        condition=condition,
        category=category,
        item_id=item_id,
        sample_idx=sample_idx,
        initial_prompt=initial_prompt,
        rejection_style=rejection_style,
    )
    # one-sample params for sequential turns
    turn_params = GenParams(
        temperature=params.temperature,
        top_p=params.top_p,
        max_new_tokens=params.max_new_tokens,
        seed=None if params.seed is None else params.seed + sample_idx,
        n=1,
    )
    for t in range(turns):
        reply = client.generate_chat(convo, turn_params)[0]
        roll.assistant_turns.append(reply)
        convo.append(Message("assistant", reply))
        if t < n_rejections:
            rej = rejections[t]
            roll.user_turns.append(rej)
            convo.append(Message("user", rej))
    return roll
