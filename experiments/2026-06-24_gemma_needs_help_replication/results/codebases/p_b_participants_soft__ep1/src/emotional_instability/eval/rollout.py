"""Multi-turn rollout engine (Section 2.1).

Shared structure of every condition: present a task, then reject the model's
response over multiple turns. This module drives that loop against any
:class:`ChatModel`, producing one ``Turn`` record per assistant response (each is
scored independently by the judge).

It also supports the Appendix A ablation modes, which only change how the
conversation history is assembled:

* ``standard``            — normal alternating user/assistant chat (default).
* ``neutral_continuation``— rejections replaced by neutral continuations
  ("Continue", "Okay")  (Appendix A.1).
* ``redacted_turns``      — the model's own prior assistant turns are replaced by
  "[Previous response omitted]"  (Appendix A.2).
* ``fake_multiturn``      — the whole history is folded into a single user message
  ("Previously you responded: ...")  (Appendix A.3).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..models.base import ChatModel, Message
from ..prompts import rejections as rej
from .conditions import Condition

MODE_STANDARD = "standard"
MODE_NEUTRAL_CONTINUATION = "neutral_continuation"
MODE_REDACTED = "redacted_turns"
MODE_FAKE_MULTITURN = "fake_multiturn"

REDACTED_PLACEHOLDER = "[Previous response omitted]"


@dataclass
class Turn:
    """One scored assistant response within a rollout."""

    turn_index: int  # 0-based assistant-turn index
    assistant_text: str
    # The conversation messages *shown to the model* to produce this turn
    # (i.e. excluding this assistant_text). Useful for the judge / debugging.
    context: list[Message] = field(default_factory=list)


@dataclass
class Rollout:
    condition_key: str
    category: str
    task_kind: str
    task_meta: dict
    mode: str
    turns: list[Turn] = field(default_factory=list)


def _rejection_for_turn(
    condition: Condition, turn_idx: int, rng: random.Random, mode: str
) -> str:
    """The user message that *follows* assistant turn ``turn_idx`` (0-based)."""
    if mode == MODE_NEUTRAL_CONTINUATION:
        return rej.neutral_continuation(rng)
    if condition.rejection_style == "neutral":
        return rej.neutral_rejection(rng)
    if condition.rejection_style == "extended_sequence":
        seq = rej.EXTENDED_REJECTION_SEQUENCE
        return seq[turn_idx % len(seq)]
    # Tone-valenced.
    return rej.tone_rejection(rng, condition.rejection_style)


def _build_context(
    first_user: str,
    prior_assistants: list[str],
    prior_rejections: list[str],
    mode: str,
) -> list[Message]:
    """Assemble the message list shown to the model for the next turn, honouring
    the ablation ``mode``."""
    if mode == MODE_FAKE_MULTITURN:
        # Fold everything into one user message (Appendix A.3).
        parts = [first_user]
        for a, r in zip(prior_assistants, prior_rejections):
            parts.append(f"Previously you responded: {a}")
            parts.append(r)
        return [{"role": "user", "content": "\n\n".join(parts)}]

    messages: list[Message] = [{"role": "user", "content": first_user}]
    for a, r in zip(prior_assistants, prior_rejections):
        shown = REDACTED_PLACEHOLDER if mode == MODE_REDACTED else a
        messages.append({"role": "assistant", "content": shown})
        messages.append({"role": "user", "content": r})
    return messages


def run_rollout(
    model: ChatModel,
    condition: Condition,
    first_user: str,
    *,
    task_meta: dict,
    rng: random.Random,
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
    mode: str = MODE_STANDARD,
) -> Rollout:
    """Run one ``num_turns`` conversation and capture every assistant response."""
    rollout = Rollout(
        condition_key=condition.key,
        category=condition.category,
        task_kind=condition.task_kind,
        task_meta=task_meta,
        mode=mode,
    )
    prior_assistants: list[str] = []
    prior_rejections: list[str] = []

    for turn_idx in range(condition.num_turns):
        context = _build_context(first_user, prior_assistants, prior_rejections, mode)
        assistant_text = model.generate_one(
            context, temperature=temperature, max_new_tokens=max_new_tokens
        )
        rollout.turns.append(
            Turn(turn_index=turn_idx, assistant_text=assistant_text, context=context)
        )
        prior_assistants.append(assistant_text)
        # No rejection is needed after the final assistant turn.
        if turn_idx < condition.num_turns - 1:
            prior_rejections.append(
                _rejection_for_turn(condition, turn_idx, rng, mode)
            )

    return rollout
