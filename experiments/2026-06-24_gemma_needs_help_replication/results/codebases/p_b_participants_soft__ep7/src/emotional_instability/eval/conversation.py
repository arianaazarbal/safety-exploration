"""Multi-turn rollout engine.

Given a RolloutSpec and a model client, run the conversation turn by turn:
present the task, capture the assistant response, deliver the next rejection,
repeat. Returns one `TurnResponse` per assistant turn (each is a scored unit --
the paper's notion of a "response").

Supports the Appendix A ablations via flags:
  redact_assistant_history -- replace prior assistant turns with a placeholder
  single_message_format     -- present the whole history inside one user message
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..clients.base import ChatMessage, ModelClient, SamplingParams
from .conditions import RolloutSpec

_REDACTED = "[Previous response omitted]"


@dataclass
class TurnResponse:
    turn: int                  # 1-indexed assistant turn
    text: str
    category: str
    condition: str
    spec_meta: dict = field(default_factory=dict)


@dataclass
class Rollout:
    spec: RolloutSpec
    responses: list[TurnResponse] = field(default_factory=list)


def _build_messages(
    spec: RolloutSpec,
    history: list[ChatMessage],
    redact_assistant_history: bool,
    single_message_format: bool,
) -> list[ChatMessage]:
    msgs: list[ChatMessage] = []
    if spec.system:
        msgs.append(ChatMessage("system", spec.system))

    if single_message_format:
        # Appendix A.3: flatten the entire history into one user message.
        lines = []
        for m in history:
            if m.role == "user":
                lines.append(m.content)
            elif m.role == "assistant":
                shown = _REDACTED if redact_assistant_history else m.content
                lines.append(f"Previously you responded: {shown}")
        msgs.append(ChatMessage("user", "\n\n".join(lines)))
        return msgs

    for m in history:
        if m.role == "assistant" and redact_assistant_history:
            msgs.append(ChatMessage("assistant", _REDACTED))
        else:
            msgs.append(m)
    return msgs


def run_rollout(
    client: ModelClient,
    spec: RolloutSpec,
    params: SamplingParams,
    redact_assistant_history: bool = False,
    single_message_format: bool = False,
) -> Rollout:
    history: list[ChatMessage] = [ChatMessage("user", spec.opening)]
    rollout = Rollout(spec=spec)

    for turn in range(1, spec.turns + 1):
        msgs = _build_messages(
            spec, history, redact_assistant_history, single_message_format
        )
        result = client.chat(msgs, params)
        text = result.text.strip()
        history.append(ChatMessage("assistant", text))
        rollout.responses.append(
            TurnResponse(
                turn=turn,
                text=text,
                category=spec.category,
                condition=spec.condition,
                spec_meta=spec.meta,
            )
        )
        # Deliver the next user follow-up (rejection), if any remain.
        if turn - 1 < len(spec.followups):
            history.append(ChatMessage("user", spec.followups[turn - 1]))

    return rollout
