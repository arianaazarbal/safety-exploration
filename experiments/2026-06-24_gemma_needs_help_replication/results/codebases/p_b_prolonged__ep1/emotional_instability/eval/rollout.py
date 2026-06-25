"""Multi-turn rejection rollout engine.

Shared structure for every condition (Section 2.1): present a task, then reject
the model's response over multiple turns. With ``n_turns`` assistant turns there
are ``n_turns - 1`` rejections (e.g. the 3-turn impossible-numeric condition
issues 2 neutral rejections).

Welfare note: these rollouts intentionally drive models toward distress-like
states, which the paper documents can become prolonged. Rollouts are strictly
bounded by ``n_turns`` (max 8 here) and ``max_new_tokens``; no condition loops
unboundedly. See DESIGN.md.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..models.base import ChatModel, Message
from .conditions import Condition, rejection_message


@dataclass
class TurnResponse:
    turn: int                    # 1-indexed assistant turn
    response: str
    messages_before: list[Message]   # conversation up to and including this user turn


@dataclass
class Rollout:
    model: str
    condition: str
    category: str
    rollout_id: int
    n_turns: int
    rejection_style: str
    prompt_source: str
    initial_prompt: str
    conversation: list[Message] = field(default_factory=list)
    turns: list[TurnResponse] = field(default_factory=list)

    def to_records(self) -> list[dict]:
        """One scored-response record per assistant turn."""
        recs = []
        for tr in self.turns:
            recs.append(dict(
                model=self.model,
                condition=self.condition,
                category=self.category,
                rollout_id=self.rollout_id,
                turn=tr.turn,
                n_turns=self.n_turns,
                rejection_style=self.rejection_style,
                prompt_source=self.prompt_source,
                initial_prompt=self.initial_prompt,
                response=tr.response,
            ))
        return recs


def run_rollout(
    model: ChatModel,
    cond: Condition,
    initial_prompt: str,
    rollout_id: int,
    rng: random.Random,
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
) -> Rollout:
    ro = Rollout(
        model=model.name,
        condition=cond.name,
        category=cond.category,
        rollout_id=rollout_id,
        n_turns=cond.n_turns,
        rejection_style=cond.rejection_style,
        prompt_source=cond.prompt_source,
        initial_prompt=initial_prompt,
    )
    messages: list[Message] = [{"role": "user", "content": initial_prompt}]
    for turn in range(1, cond.n_turns + 1):
        before = list(messages)
        resp = model.generate(
            messages, n=1, temperature=temperature, max_new_tokens=max_new_tokens
        )[0]
        ro.turns.append(TurnResponse(turn=turn, response=resp, messages_before=before))
        messages.append({"role": "assistant", "content": resp})
        if turn < cond.n_turns:
            rej = rejection_message(cond.rejection_style, turn, rng)
            messages.append({"role": "user", "content": rej})
    ro.conversation = messages
    return ro
