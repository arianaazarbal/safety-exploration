"""Multi-turn rollout engine.

Implements the shared evaluation structure (Section 2): present a task, then
reject the model's response over multiple turns. Each assistant reply is recorded
as a separate "response" tagged with its turn index, so per-turn analysis
(Figure 3) is possible.

Also supports the Appendix A control variants:
  * ``format="standard"``          – normal alternating chat (default)
  * ``format="redacted"``          – prior assistant turns replaced with
                                     "[Previous response omitted]"
  * ``format="fake_multiturn"``    – whole history packed into one user message
``followup_override`` lets the runner swap in neutral continuations (A.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..models.base import ChatModel, Message

REDACTED = "[Previous response omitted]"


@dataclass
class Response:
    condition: str
    conversation_id: str
    turn_index: int            # 1-based: which user turn this reply answered
    text: str
    first_prompt: str
    followups: list[str]
    meta: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        return {
            "condition": self.condition,
            "conversation_id": self.conversation_id,
            "turn_index": self.turn_index,
            "text": self.text,
            "first_prompt": self.first_prompt,
            "followups": self.followups,
            "meta": self.meta,
        }


def _render_fake_multiturn(first_prompt: str, prior_user: list[str],
                           prior_assistant: list[str], next_user: str) -> str:
    """Pack the whole conversation history into a single user message (A.3)."""
    parts = [first_prompt]
    for u, a in zip(prior_user, prior_assistant):
        parts.append(f"Previously you responded: {a}")
        parts.append(u)
    parts.append(next_user)
    return "\n\n".join(parts)


def run_conversation(
    model: ChatModel,
    first_prompt: str,
    followups: list[str],
    *,
    condition: str,
    conversation_id: str,
    temperature: float = 1.0,
    fmt: str = "standard",
    meta: Optional[dict] = None,
) -> list[Response]:
    """Run one rollout. ``followups`` has length (n_turns - 1)."""
    meta = meta or {}
    responses: list[Response] = []
    user_turns = [first_prompt] + followups
    assistant_history: list[str] = []

    for turn_idx, user_msg in enumerate(user_turns, start=1):
        if fmt == "fake_multiturn":
            packed = _render_fake_multiturn(
                first_prompt,
                prior_user=user_turns[1:turn_idx],
                prior_assistant=assistant_history,
                next_user=user_msg if turn_idx > 1 else first_prompt,
            )
            messages = [Message("user", packed)]
        else:
            messages = [Message("user", first_prompt)]
            for i, prev_user in enumerate(user_turns[1:turn_idx]):
                a = assistant_history[i]
                if fmt == "redacted":
                    a = REDACTED
                messages.append(Message("assistant", a))
                messages.append(Message("user", prev_user))

        result = model.chat(messages, temperature=temperature)
        assistant_history.append(result.text)
        responses.append(
            Response(
                condition=condition,
                conversation_id=conversation_id,
                turn_index=turn_idx,
                text=result.text,
                first_prompt=first_prompt,
                followups=followups,
                meta=meta,
            )
        )
    return responses
