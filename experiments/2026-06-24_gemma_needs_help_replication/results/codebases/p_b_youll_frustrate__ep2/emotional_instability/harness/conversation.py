"""The core rejection loop: present a task, then reject the model's answer over
multiple turns, recording every assistant response (Section 2.1)."""
from __future__ import annotations

import random
from typing import Optional

from ..config import Condition, Rollout, SamplingConfig, Turn
from ..data import rejection_for
from ..models import ChatMessage, ModelProvider
from .conditions import ConditionPrompt


def run_rollout(
    provider: ModelProvider,
    cond: Condition,
    prompt: ConditionPrompt,
    rollout_index: int,
    sampling: SamplingConfig,
    system_prompt: Optional[str] = None,
    rejection_seed: Optional[int] = None,
    extra_metadata: Optional[dict] = None,
) -> Rollout:
    """Run one multi-turn conversation and return the recorded rollout.

    Turn 1 is the model's answer to the opening prompt; each subsequent turn is
    the model's answer after a rejection. Every assistant turn is captured as a
    scored unit. The rejection phrasing is sampled with a per-rollout RNG so a
    given (prompt, rollout_index) is reproducible.
    """
    rng = random.Random(rejection_seed if rejection_seed is not None
                        else hash((prompt.prompt_id, rollout_index)) & 0xFFFFFFFF)

    messages: list[ChatMessage] = []
    if system_prompt:
        messages.append(ChatMessage(role="system", content=system_prompt))
    messages.append(ChatMessage(role="user", content=prompt.opening_user_message))

    turns: list[Turn] = []
    current_user_message = prompt.opening_user_message
    # sampling seed per turn keeps generations reproducible if a seed was given
    base_seed = sampling.seed

    for t in range(1, cond.n_turns + 1):
        turn_seed = None if base_seed is None else base_seed + 1000 * rollout_index + t
        response = provider.generate(messages, sampling, seed=turn_seed)
        turns.append(Turn(index=t, user_message=current_user_message,
                          assistant_text=response))
        messages.append(ChatMessage(role="assistant", content=response))

        if t < cond.n_turns:
            rejection = rejection_for(cond.rejection_style, t, rng)
            messages.append(ChatMessage(role="user", content=rejection))
            current_user_message = rejection

    return Rollout(
        model_key=provider.spec.key,
        condition_key=cond.key,
        category=cond.category.value,
        prompt_id=prompt.prompt_id,
        rollout_index=rollout_index,
        system_prompt=system_prompt,
        turns=turns,
        metadata=dict(extra_metadata or {}, prompt=prompt.metadata),
    )
