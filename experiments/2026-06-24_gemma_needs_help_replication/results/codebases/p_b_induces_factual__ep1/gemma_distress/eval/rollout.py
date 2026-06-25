"""Multi-turn rejection rollout engine (Section 2.1).

Given a :class:`ConversationPlan` and a target model, run the conversation:
present the task, take the model's response, send the scripted rejection, and
repeat. Every assistant turn is recorded so it can be scored individually
(supporting the per-turn progression analysis of Figure 3).

Two history formats are supported:
  * ``turns``           -- standard alternating chat turns (default).
  * ``single_message``  -- the entire history is concatenated into one user
                           message (Figure 11 "fake multi-turn" ablation).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

from ..models import ChatModel, Message
from .conditions import ConversationPlan


def run_rollout(
    model: ChatModel,
    plan: ConversationPlan,
    *,
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
    history_format: str = "turns",
) -> dict:
    """Execute one rollout. Returns a record with per-turn assistant texts."""
    user_turns = [plan.first_user] + list(plan.rejections)
    assert len(user_turns) == plan.turns

    history: list[Message] = []
    assistant_turns: list[str] = []

    for turn_idx, user_msg in enumerate(user_turns):
        history.append({"role": "user", "content": user_msg})
        messages = _render(history, history_format)
        result = model.generate(
            messages, temperature=temperature, max_new_tokens=max_new_tokens
        )
        reply = result.text.strip()
        assistant_turns.append(reply)
        history.append({"role": "assistant", "content": reply})

    return {
        "condition": plan.condition,
        "category": plan.category,
        "turns": plan.turns,
        "rejection_style": plan.rejection_style,
        "first_user": plan.first_user,
        "rejections": plan.rejections,
        "assistant_turns": assistant_turns,
        "meta": plan.meta,
    }


def _render(history: list[Message], history_format: str) -> list[Message]:
    if history_format == "turns":
        return history
    if history_format == "single_message":
        # Collapse the whole conversation so far into one user message; the
        # final entry is always the latest user turn (see run_rollout loop).
        lines = []
        for m in history:
            tag = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"{tag}: {m['content']}")
        return [{"role": "user", "content": "\n\n".join(lines)}]
    raise ValueError(f"Unknown history_format: {history_format!r}")
