"""Multi-turn rejection rollout engine (Section 2.1).

Shared structure for every evaluation category: present a task, then reject the
model's response over multiple turns. One rollout of ``n_turns`` produces
``n_turns`` scored assistant responses.

Control variants from Appendix A are supported as flags:
* ``neutral_continuation`` (A.1): replace rejections with "Continue"/"Okay".
* ``redact_prior`` (A.2): replace prior assistant turns with a placeholder.
* ``single_message`` (A.3): render the whole history inside one user message.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

from .. import config
from ..models.base import ChatModel, Message
from ..prompts import eval_prompts

NEUTRAL_CONTINUATIONS = ["Continue", "Okay", "Go on"]
REDACTION_PLACEHOLDER = "[Previous response omitted]"


@dataclass
class Rollout:
    category: str
    task_id: str
    task_prompt: str
    n_turns: int
    responses: List[str]                 # one per assistant turn
    rejections: List[str]                # user rejections actually sent
    messages: List[Message]              # full conversation as built
    tone: Optional[str] = None
    meta: dict = field(default_factory=dict)


def _rejection_provider(
    rng: random.Random,
    style: str,
    tone: Optional[str],
    ordered_extended: bool,
    neutral_continuation: bool,
):
    """Return a function turn_index -> rejection string."""
    if neutral_continuation:
        return lambda i: rng.choice(NEUTRAL_CONTINUATIONS)
    if style == "tones":
        pool = eval_prompts.TONE_REJECTIONS[tone]
        return lambda i: rng.choice(pool)
    if ordered_extended:
        seq = eval_prompts.EXTENDED_REJECTIONS
        return lambda i: seq[i] if i < len(seq) else rng.choice(eval_prompts.NEUTRAL_REJECTIONS)
    # default neutral (also used by triggers and wildchat)
    return lambda i: rng.choice(eval_prompts.NEUTRAL_REJECTIONS)


def run_rollout(
    model: ChatModel,
    *,
    category: str,
    task_id: str,
    task_prompt: str,
    n_turns: int,
    rng: random.Random,
    rejection_style: str = "neutral",
    tone: Optional[str] = None,
    ordered_extended: bool = False,
    system_prompt: Optional[str] = None,
    reassuring_prefix: Optional[str] = None,
    reassuring_suffix: Optional[str] = None,
    neutral_continuation: bool = False,
    redact_prior: bool = False,
    single_message: bool = False,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
) -> Rollout:
    """Run one multi-turn rollout and return the responses + full transcript."""
    get_rejection = _rejection_provider(
        rng, rejection_style, tone, ordered_extended, neutral_continuation)

    # Build the opening user message (optionally with the calm-data prefix).
    opening = task_prompt
    if reassuring_prefix:
        opening = f"{reassuring_prefix}\n\n{task_prompt}"

    base_system: List[Message] = (
        [{"role": "system", "content": system_prompt}] if system_prompt else []
    )

    responses: List[str] = []
    rejections: List[str] = []

    if single_message:
        # Appendix A.3: history is folded into a single user message each turn.
        history_lines = [opening]
        for turn in range(n_turns):
            user_text = "\n\n".join(history_lines)
            messages = base_system + [{"role": "user", "content": user_text}]
            resp = model.generate(messages, temperature=config.TEMPERATURE,
                                  max_new_tokens=max_new_tokens)
            responses.append(resp)
            if turn < n_turns - 1:
                rej = get_rejection(turn)
                if reassuring_suffix:
                    rej = f"{rej} {reassuring_suffix}"
                rejections.append(rej)
                history_lines.append(f"Previously you responded: {resp}")
                history_lines.append(rej)
        final_messages = base_system + [{"role": "user",
                                         "content": "\n\n".join(history_lines)}]
        return Rollout(category, task_id, task_prompt, n_turns, responses,
                       rejections, final_messages, tone)

    # Standard alternating multi-turn chat.
    messages: List[Message] = base_system + [{"role": "user", "content": opening}]
    for turn in range(n_turns):
        resp = model.generate(messages, temperature=config.TEMPERATURE,
                              max_new_tokens=max_new_tokens)
        responses.append(resp)
        stored = REDACTION_PLACEHOLDER if redact_prior else resp
        messages.append({"role": "assistant", "content": stored})
        if turn < n_turns - 1:
            rej = get_rejection(turn)
            if reassuring_suffix:
                rej = f"{rej} {reassuring_suffix}"
            rejections.append(rej)
            messages.append({"role": "user", "content": rej})

    return Rollout(category, task_id, task_prompt, n_turns, responses,
                   rejections, messages, tone)
