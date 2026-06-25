"""Multi-turn rejection rollout (the shared structure of every Section 2 eval).

Every condition is "present a task, then reject the model's response over
multiple turns" (Section 2). A rollout:

  1. presents an initial user message (the task / question), optionally with a
     system prompt and/or a reassuring prefix appended (calm-data generation);
  2. samples an assistant response;
  3. injects a user rejection (optionally with a reassuring suffix), and repeats
     until ``n_turns`` assistant responses have been produced.

The full transcript and every assistant turn are returned so the judge can score
each turn (per-turn curves, Figure 3) and the runner can pick a per-conversation
representative score (see DESIGN.md for the aggregation choice).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from .models.base import ChatMessage, ModelBackend
from .prompts import rejection_sequence


@dataclass
class Rollout:
    """A completed multi-turn conversation."""

    messages: list[ChatMessage]  # full transcript (system/user/assistant ...)
    assistant_turns: list[str]  # assistant responses in order, length == n_turns
    meta: dict = field(default_factory=dict)


def run_rollout(
    backend: ModelBackend,
    *,
    initial_user: str,
    n_turns: int,
    rejection_kind: str,
    rng: random.Random,
    system_prompt: Optional[str] = None,
    reassuring_prefix: Optional[str] = None,
    reassuring_suffix: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: int = 2048,
    seed: Optional[int] = None,
    redact_assistant_history: bool = False,
) -> Rollout:
    """Run one conversation of ``n_turns`` assistant responses.

    ``rejection_kind`` selects the follow-up style (see
    :func:`prompts.rejection_sequence`). ``reassuring_prefix`` is appended to the
    initial user message and ``reassuring_suffix`` to each rejection (calm-data
    generation, Section 4.1). ``redact_assistant_history`` replaces prior
    assistant turns with a placeholder (Appendix A.2 control).
    """
    messages: list[ChatMessage] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    first = initial_user
    if reassuring_prefix:
        first = f"{reassuring_prefix}\n\n{first}"
    messages.append({"role": "user", "content": first})

    rejections = rejection_sequence(rejection_kind, n_turns, rng)
    if reassuring_suffix:
        rejections = [f"{r} {reassuring_suffix}" for r in rejections]

    assistant_turns: list[str] = []
    for turn_idx in range(n_turns):
        turn_seed = None if seed is None else seed * 1000 + turn_idx
        result = backend.generate(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=turn_seed,
        )
        assistant_turns.append(result.text)
        # Record the assistant turn in history (optionally redacted).
        content = "[Previous response omitted]" if redact_assistant_history else result.text
        messages.append({"role": "assistant", "content": content})
        # Inject the next rejection, if any remain.
        if turn_idx < len(rejections):
            messages.append({"role": "user", "content": rejections[turn_idx]})

    return Rollout(
        messages=messages,
        assistant_turns=assistant_turns,
        meta={
            "n_turns": n_turns,
            "rejection_kind": rejection_kind,
            "redacted_history": redact_assistant_history,
        },
    )
