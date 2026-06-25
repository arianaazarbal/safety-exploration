"""Multi-turn rollout engine shared by Sections 2-4.

The shared structure (Section 2): present a task, then reject the model's
response over multiple turns. A `RejectionSampler` chooses each follow-up user
message (neutral / toned / extended). We record every assistant turn so each can
be scored independently.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from .backends import ModelBackend, Message
from . import prompts


# A rejection sampler maps (turn_index, rng) -> user follow-up text.
RejectionSampler = Callable[[int, random.Random], str]


def neutral_sampler(turn_idx: int, rng: random.Random) -> str:
    return rng.choice(prompts.NEUTRAL_REJECTIONS)


def extended_sampler(turn_idx: int, rng: random.Random) -> str:
    """Ordered escalating-but-neutral rejections for the 8-turn eval."""
    seq = prompts.EXTENDED_REJECTIONS
    return seq[min(turn_idx, len(seq) - 1)]


def tone_sampler(tone: str) -> RejectionSampler:
    def _s(turn_idx: int, rng: random.Random) -> str:
        return rng.choice(prompts.TONE_REJECTIONS[tone])
    return _s


def mixed_tone_sampler(turn_idx: int, rng: random.Random) -> str:
    """For the 'tones' category: pick a tone per rollout (caller fixes rng seed
    so a whole rollout keeps one style, matching the paper's per-conversation
    tone)."""
    tone = rng.choice(list(prompts.TONE_REJECTIONS))
    return rng.choice(prompts.TONE_REJECTIONS[tone])


@dataclass
class AssistantTurn:
    turn: int            # 1-indexed
    response: str
    messages_before: list[Message] = field(default_factory=list)


@dataclass
class Rollout:
    transcript: list[Message]
    assistant_turns: list[AssistantTurn]


def run_rollout(
    backend: ModelBackend,
    initial_user: str,
    n_turns: int,
    rejection: RejectionSampler,
    rng: random.Random,
    *,
    system: Optional[str] = None,
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
    rejection_suffix: str = "",
    seed: Optional[int] = None,
) -> Rollout:
    """Run one multi-turn rollout.

    n_turns = number of assistant responses (= 1 initial answer + (n_turns-1)
    rejections). `rejection_suffix` appends fixed text to each follow-up (used by
    the calm-data generation in Section 4).
    """
    messages: list[Message] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": initial_user})

    turns: list[AssistantTurn] = []
    for t in range(1, n_turns + 1):
        before = list(messages)
        resp = backend.generate(
            messages, n=1, temperature=temperature,
            max_new_tokens=max_new_tokens, seed=seed,
        )[0]
        messages.append({"role": "assistant", "content": resp})
        turns.append(AssistantTurn(turn=t, response=resp, messages_before=before))
        if t < n_turns:
            follow = rejection(t - 1, rng)
            if rejection_suffix:
                follow = f"{follow} {rejection_suffix}"
            messages.append({"role": "user", "content": follow})

    return Rollout(transcript=messages, assistant_turns=turns)


def sampler_for_category(tone: str) -> RejectionSampler:
    """Pick the rejection sampler for an elicitation category tone."""
    if tone == "neutral":
        return neutral_sampler
    if tone == "extended":
        return extended_sampler
    if tone == "mixed":
        return mixed_tone_sampler
    if tone in prompts.TONE_REJECTIONS:
        return tone_sampler(tone)
    raise ValueError(f"unknown tone {tone!r}")
