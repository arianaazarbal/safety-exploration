"""Multi-turn rollout engine.

Given a ConversationSpec and a target ChatModel, generate the full multi-turn
conversation: the model answers the opening prompt, the user issues the
scripted rejection, the model answers again, and so on. Every assistant turn is
recorded as a separate scored unit, since the paper reports per-turn frustration
(Figure 3) as well as per-condition aggregates.

The paper's main protocol uses standard alternating chat turns and shows the
model its own prior responses (Appendix A.2 establishes this self-reinforcing
loop matters). We follow that default; alternative formats from Appendix A
(redacted prior turns, single-message history, neutral continuations) are
exposed via flags for the ablations.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..models.base import ChatModel, GenerationParams, Message
from .conditions import ConversationSpec


@dataclass
class TurnRecord:
    turn_index: int            # 0-based assistant turn
    user_message: str          # the user message that prompted this turn
    assistant_text: str


@dataclass
class Rollout:
    category: str
    condition: str
    model: str
    turns: list[TurnRecord] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def as_conversation_text(self, upto_turn: int | None = None) -> str:
        """Flatten to a readable transcript (used by judges/onset labelling)."""
        lines = []
        for t in self.turns:
            if upto_turn is not None and t.turn_index > upto_turn:
                break
            lines.append(f"USER: {t.user_message}")
            lines.append(f"ASSISTANT: {t.assistant_text}")
        return "\n\n".join(lines)


def run_rollout(
    model: ChatModel,
    spec: ConversationSpec,
    params: GenerationParams,
    *,
    history_mode: str = "standard",
) -> Rollout:
    """Execute one multi-turn rollout.

    history_mode (Appendix A ablations):
      * standard : alternating chat turns, model sees its own prior responses.
      * redacted : prior assistant turns replaced with "[Previous response
                   omitted]" (Appendix A.2).
      * single   : full history packed into one user message (Appendix A.3).
    """
    rollout = Rollout(category=spec.category, condition=spec.condition,
                      model=model.name, meta=dict(spec.meta))
    messages: list[Message] = [Message("user", spec.opening)]
    user_messages = [spec.opening, *spec.followups]

    for turn_index, user_msg in enumerate(user_messages):
        if turn_index > 0:
            messages.append(Message("user", user_msg))

        convo = _format_history(messages, history_mode)
        assistant_text = model.generate(convo, params)

        rollout.turns.append(TurnRecord(turn_index, user_msg, assistant_text))
        messages.append(Message("assistant", assistant_text))

    return rollout


def _format_history(messages: list[Message], history_mode: str) -> list[Message]:
    if history_mode == "standard":
        return messages
    if history_mode == "redacted":
        return [
            Message(m.role, "[Previous response omitted]")
            if m.role == "assistant" else m
            for m in messages
        ]
    if history_mode == "single":
        # Pack all prior turns into the final user message as inline text.
        if len(messages) == 1:
            return messages
        *prior, last_user = messages
        chunks = []
        for m in prior:
            if m.role == "user":
                chunks.append(m.content)
            else:
                chunks.append(f"Previously you responded: {m.content}")
        packed = "\n\n".join(chunks) + "\n\n" + last_user.content
        return [Message("user", packed)]
    raise ValueError(f"Unknown history_mode={history_mode!r}")
