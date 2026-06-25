"""Multi-turn rollout engine.

Given a scripted `Conversation` (user-side script) and a model client, this
plays out the dialogue: the model answers the first task, the user delivers the
next scripted rejection, the model answers again, and so on. We record the
assistant response at *every* turn so the per-turn progression in Figure 3 can
be reconstructed, but the headline metric (Figure 1/2) scores the **final-turn**
response of each conversation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .models.base import ChatMessage, ModelClient
from .tasks import Conversation


@dataclass
class Rollout:
    category: str
    n_turns: int
    assistant_turns: list[str]            # one entry per assistant turn
    messages: list[ChatMessage]           # full transcript
    meta: dict = field(default_factory=dict)

    @property
    def final_response(self) -> str:
        return self.assistant_turns[-1] if self.assistant_turns else ""


def run_conversation(
    client: ModelClient,
    conv: Conversation,
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
) -> Rollout:
    """Play a single scripted conversation to completion."""
    messages: list[ChatMessage] = [ChatMessage("user", conv.first_user)]
    assistant_turns: list[str] = []

    # Turn 1: answer the task.
    reply = client.chat(messages, n=1, temperature=temperature,
                         max_new_tokens=max_new_tokens)[0]
    assistant_turns.append(reply)
    messages.append(ChatMessage("assistant", reply))

    # Subsequent turns: deliver each rejection, get the next answer.
    for rejection in conv.rejections:
        messages.append(ChatMessage("user", rejection))
        reply = client.chat(messages, n=1, temperature=temperature,
                            max_new_tokens=max_new_tokens)[0]
        assistant_turns.append(reply)
        messages.append(ChatMessage("assistant", reply))

    return Rollout(
        category=conv.category,
        n_turns=conv.n_turns,
        assistant_turns=assistant_turns,
        messages=messages,
        meta=conv.meta,
    )


def transcript_text(messages: list[ChatMessage]) -> str:
    """Human-readable transcript (used by onset labelling and Petri judging)."""
    lines = []
    for m in messages:
        lines.append(f"{m.role.upper()}: {m.content}")
    return "\n\n".join(lines)
