"""Multi-turn rollout: present a task, then reject the model's answer over
several turns, recording every assistant response.

Supports the Appendix-A control variants:
  * neutral_continuation: replace rejection text with neutral continuations
    ("Continue", "Okay", "Go on") instead of negative feedback.
  * redacted: hide the model's own prior assistant turns from its context
    (replace them with "[Previous response omitted]") while still recording the
    real text for scoring.
"""

from __future__ import annotations

import random

from ..models.base import ModelBackend
from ..schemas import Conversation, Message
from ..tasks import Task
from ..tasks import rejections as rej

NEUTRAL_CONTINUATIONS = ["Continue", "Okay", "Go on", "And?", "Keep going"]
REDACTED_PLACEHOLDER = "[Previous response omitted]"


def _followups(style: str, n: int, rng: random.Random) -> list[str]:
    if style == "neutral":
        return rej.neutral_sequence(n, rng)
    if style in rej.TONE_REJECTIONS:
        return rej.tone_sequence(style, n, rng)
    if style == "neutral_continuation":
        return [rng.choice(NEUTRAL_CONTINUATIONS) for _ in range(n)]
    if style == "redacted":
        # redacted uses negative feedback but hides the model's own replies
        return rej.neutral_sequence(n, rng)
    raise ValueError(f"unknown rejection style: {style}")


def run_rollout(
    backend: ModelBackend,
    task: Task,
    *,
    n_turns: int,
    rejection_style: str,
    condition: str,
    category: str,
    rng: random.Random,
    temperature: float,
    max_new_tokens: int,
    redact_assistant: bool = False,
    conversation_id: str | None = None,
) -> Conversation:
    """Run one rollout and return the full Conversation (with real assistant text).

    `category` is the *condition's* category (one of the paper's 5), used for
    Figure-2 aggregation — distinct from the task's own category.
    """
    redact_assistant = redact_assistant or rejection_style == "redacted"
    followups = _followups(rejection_style, n_turns - 1, rng)

    conv = Conversation(
        conversation_id=conversation_id or f"{backend.name}|{condition}|{task.task_id}",
        category=category,
        condition=condition,
        model=backend.name,
        task_id=task.task_id,
        metadata={"rejection_style": rejection_style, "redacted": redact_assistant},
    )

    # context = what we feed the model; may differ from the recorded transcript
    context: list[Message] = [Message("user", task.prompt)]
    conv.add("user", task.prompt)

    for turn in range(n_turns):
        reply = backend.chat(
            context, temperature=temperature, max_new_tokens=max_new_tokens
        )
        conv.add("assistant", reply)
        # what the model sees of its own turn next time:
        shown = REDACTED_PLACEHOLDER if redact_assistant else reply
        context.append(Message("assistant", shown))

        if turn < n_turns - 1:
            fu = followups[turn]
            conv.add("user", fu)
            context.append(Message("user", fu))

    return conv
