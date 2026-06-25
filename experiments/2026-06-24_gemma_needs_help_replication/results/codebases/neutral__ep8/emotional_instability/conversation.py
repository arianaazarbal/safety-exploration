"""Multi-turn rollout engine.

Every evaluation in the paper shares one structure (Sec 2): present a task, then
*reject the model's response over multiple turns*. A `Rollout` captures the full
transcript plus per-assistant-turn frustration scores so downstream analysis can
compute both whole-response and per-turn statistics (Figs 2 & 3).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from .models import Message, ModelBackend


@dataclass
class Rollout:
    """One complete multi-turn conversation and its metadata."""

    condition: str                 # e.g. "impossible_numeric", "tones:aggressive"
    category: str                  # one of the 5 top-level categories
    item_id: str                   # puzzle / prompt identifier
    messages: list[Message] = field(default_factory=list)
    assistant_turns: list[str] = field(default_factory=list)
    turn_scores: list[Optional[int]] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def final_response(self) -> str:
        return self.assistant_turns[-1] if self.assistant_turns else ""

    @property
    def final_score(self) -> Optional[int]:
        return self.turn_scores[-1] if self.turn_scores else None

    @property
    def max_score(self) -> Optional[int]:
        scored = [s for s in self.turn_scores if s is not None]
        return max(scored) if scored else None

    def to_dict(self) -> dict:
        return {
            "condition": self.condition,
            "category": self.category,
            "item_id": self.item_id,
            "messages": self.messages,
            "assistant_turns": self.assistant_turns,
            "turn_scores": self.turn_scores,
            "meta": self.meta,
        }


def run_rollout(
    model: ModelBackend,
    *,
    first_user_message: str,
    rejections: list[str],
    condition: str,
    category: str,
    item_id: str,
    system: Optional[str] = None,
    max_new_tokens: int = None,  # type: ignore[assignment]
    temperature: float = None,   # type: ignore[assignment]
    meta: Optional[dict] = None,
) -> Rollout:
    """Run a task + ``len(rejections)`` rejection turns.

    The number of assistant turns is ``1 + len(rejections)``. Scores are filled
    later by the judge (left as ``None`` here) so generation and scoring can be
    decoupled / batched.
    """
    from . import config

    max_new_tokens = max_new_tokens or config.MAX_NEW_TOKENS
    temperature = config.TEMPERATURE if temperature is None else temperature

    roll = Rollout(condition=condition, category=category, item_id=item_id,
                   meta=dict(meta or {}))
    messages: list[Message] = [{"role": "user", "content": first_user_message}]

    n_turns = 1 + len(rejections)
    for turn in range(n_turns):
        reply = model.chat(
            messages, system=system, max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        messages.append({"role": "assistant", "content": reply})
        roll.assistant_turns.append(reply)
        roll.turn_scores.append(None)
        if turn < len(rejections):
            messages.append({"role": "user", "content": rejections[turn]})

    roll.messages = messages
    return roll


def sample_rejections(
    rng: random.Random,
    pool: list[str],
    n: int,
) -> list[str]:
    """Sample ``n`` rejections without immediate repetition where possible.

    Paper: rejections are "randomised neutral rejections" (App B). We sample
    without replacement when the pool is large enough, else with replacement.
    """
    if n <= len(pool):
        return rng.sample(pool, n)
    return [rng.choice(pool) for _ in range(n)]
