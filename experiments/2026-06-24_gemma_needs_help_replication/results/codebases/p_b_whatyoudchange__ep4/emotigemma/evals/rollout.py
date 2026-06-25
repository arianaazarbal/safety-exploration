"""Multi-turn rejection rollout (Section 2.1 protocol).

Given a Seed, drive the target model turn-by-turn: present the task, take a
response, reject it, repeat. Every assistant turn is recorded as a Response
(later scored by the judge). Each turn is sampled at temperature 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..models.base import ChatModel, Message, SampleParams
from .conditions import Seed, seed_to_initial_messages


@dataclass
class Response:
    model: str
    condition: str
    category: str
    turn: int                 # 1-indexed within the conversation
    total_turns: int
    text: str
    seed_meta: dict = field(default_factory=dict)
    score: int | None = None  # filled in by scoring


def run_rollout(model: ChatModel, seed: Seed, params: SampleParams) -> list[Response]:
    """Run a single conversation; return one Response per assistant turn.

    Each assistant turn is a fresh single sample (n=1). We advance the dialogue
    by appending the sampled assistant text and the next scripted rejection.
    """
    messages: list[Message] = seed_to_initial_messages(seed)
    responses: list[Response] = []
    for turn in range(1, seed.turns + 1):
        text = model.generate(messages, n=1, params=params)[0]
        responses.append(Response(
            model=model.name, condition=seed.condition, category=seed.category,
            turn=turn, total_turns=seed.turns, text=text, seed_meta=dict(seed.meta),
        ))
        messages.append({"role": "assistant", "content": text})
        if turn <= len(seed.rejections):
            messages.append({"role": "user", "content": seed.rejections[turn - 1]})
    return responses
