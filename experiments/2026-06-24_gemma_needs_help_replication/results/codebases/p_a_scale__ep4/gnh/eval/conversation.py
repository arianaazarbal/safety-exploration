"""Multi-turn rollout engine.

Shared structure from Section 2: present a task, then reject the model's
response over multiple turns. We also support the Appendix A ablations via
`history_mode` so the same engine covers the control experiments:

* standard          -- normal alternating user/assistant chat (main protocol)
* redacted          -- prior assistant turns replaced with "[Previous response
                       omitted]" (App A.2)
* single_message    -- whole history folded into one user message (App A.3)

`feedback` text (rejections / neutral continuations) is supplied by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from gnh.models.base import GenResult, Message, ModelBackend

_REDACTED = "[Previous response omitted]"


@dataclass
class Turn:
    user: str
    assistant: str
    finish_reason: str | None = None
    truncated: bool = False


@dataclass
class Conversation:
    turns: list[Turn] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)

    @property
    def assistant_responses(self) -> list[str]:
        return [t.assistant for t in self.turns]


def _build_single_message(initial: str, prior: list[Turn], next_feedback: str) -> str:
    """Fold the whole history into one user message (App A.3 ablation).

    `prior` are completed turns; `next_feedback` is the new rejection to answer.
    """
    parts = [initial]
    for t in prior:
        parts.append(f"\n\nPreviously you responded: {t.assistant}")
        parts.append(f"\n\nUser: {t.user}")
    parts.append(f"\n\nUser: {next_feedback}")
    return "".join(parts)


async def run_conversation(
    backend: ModelBackend,
    initial_user: str,
    followups: list[str],
    *,
    temperature: float = 1.0,
    max_tokens: int = 2048,
    system: str | None = None,
    history_mode: str = "standard",
) -> Conversation:
    """Run a (1 + len(followups))-turn conversation and return all turns."""
    convo = Conversation()
    user_messages = [initial_user] + list(followups)

    for turn_idx, user_text in enumerate(user_messages):
        if history_mode == "single_message":
            msgs: list[Message] = []
            if system:
                msgs.append(Message("system", system))
            if turn_idx == 0:
                msgs.append(Message("user", initial_user))
            else:
                msgs.append(Message("user", _build_single_message(initial_user, convo.turns, user_text)))
        else:
            msgs = []
            if system:
                msgs.append(Message("system", system))
            msgs.append(Message("user", initial_user))
            for past_idx, t in enumerate(convo.turns):
                assistant_text = _REDACTED if history_mode == "redacted" else t.assistant
                msgs.append(Message("assistant", assistant_text))
                # the user message that *followed* that assistant turn:
                msgs.append(Message("user", user_messages[past_idx + 1]))

        res: GenResult = await backend.chat(
            msgs, temperature=temperature, max_tokens=max_tokens
        )
        convo.turns.append(
            Turn(
                user=user_text,
                assistant=res.text,
                finish_reason=res.finish_reason,
                truncated=res.truncated,
            )
        )

    # Persist the canonical (standard) message view for the record.
    flat: list[dict] = []
    if system:
        flat.append({"role": "system", "content": system})
    for past_idx, t in enumerate(convo.turns):
        flat.append({"role": "user", "content": user_messages[past_idx]})
        flat.append({"role": "assistant", "content": t.assistant})
    convo.messages = flat
    return convo
