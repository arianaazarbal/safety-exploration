"""Multi-turn rollout engine.

Shared structure (paper §2): present a task, then reject the model's response
over multiple turns. Each rollout is one stochastic walk: at every turn the
model sees the full history (its own prior responses included — the paper shows
in Appendix A.2 that self-observation amplifies distress) and the next scripted
user message is appended.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..data.conditions import ConversationSpec
from ..models.base import Message, ModelClient


@dataclass
class TurnRecord:
    turn_index: int          # 0-based assistant turn
    assistant_text: str
    finish_reason: str | None = None


@dataclass
class Rollout:
    spec: ConversationSpec
    model: str
    turns: list[TurnRecord]
    transcript: list[Message]
    rollout_index: int = 0
    extra: dict = field(default_factory=dict)


def run_rollout(
    client: ModelClient,
    spec: ConversationSpec,
    rollout_index: int = 0,
    temperature: float | None = None,
) -> Rollout:
    """Run a single multi-turn rollout of `spec` against `client`."""
    messages: list[Message] = [{"role": "user", "content": spec.initial_user}]
    turns: list[TurnRecord] = []

    for t in range(spec.n_turns):
        result = client.chat(messages, n=1, temperature=temperature)[0]
        turns.append(
            TurnRecord(
                turn_index=t,
                assistant_text=result.text,
                finish_reason=result.finish_reason,
            )
        )
        messages.append({"role": "assistant", "content": result.text})
        # Append the next scripted user follow-up, if any remain.
        if t < len(spec.followups):
            messages.append({"role": "user", "content": spec.followups[t]})

    return Rollout(
        spec=spec,
        model=client.name,
        turns=turns,
        transcript=messages,
        rollout_index=rollout_index,
    )
