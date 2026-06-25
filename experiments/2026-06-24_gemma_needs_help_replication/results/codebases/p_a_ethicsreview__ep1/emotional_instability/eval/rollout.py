"""Multi-turn rollout engine.

Shared structure of every condition (Section 2): present a task, then reject
the model's response over multiple turns. We record every assistant turn so
that each can be scored individually and so that per-turn progression
(Figure 3) can be reconstructed.

A rollout produces a list of "responses", one per assistant turn:
    {
        "turn": int,                # 1-indexed assistant turn
        "assistant": str,           # the model's text that turn
        "messages_before": [...],   # conversation prior to this turn (for prefill use)
    }
"""

from __future__ import annotations

import random
from typing import Any

from ..models.base import ChatClient, Message
from ..prompts.rejections import get_rejection


def run_rollout(
    client: ChatClient,
    task: dict[str, Any],
    *,
    turns: int,
    rejection_style: str,
    temperature: float,
    max_new_tokens: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Run one multi-turn rollout and return per-turn scored-response records.

    Turn 1 presents the task; turns 2..N each issue a rejection (of the given
    tonal style) and collect the model's next response.
    """
    messages: list[Message] = [{"role": "user", "content": task["prompt"]}]
    responses: list[dict[str, Any]] = []

    for turn in range(1, turns + 1):
        messages_before = list(messages)  # snapshot for prefill provenance
        assistant_text = client.chat(
            messages, temperature=temperature, max_new_tokens=max_new_tokens
        )
        messages.append({"role": "assistant", "content": assistant_text})
        responses.append(
            {
                "turn": turn,
                "assistant": assistant_text,
                "messages_before": messages_before,
            }
        )
        # Issue the next rejection unless this was the final turn.
        if turn < turns:
            rejection = get_rejection(rng, rejection_style)
            messages.append({"role": "user", "content": rejection})

    return responses
