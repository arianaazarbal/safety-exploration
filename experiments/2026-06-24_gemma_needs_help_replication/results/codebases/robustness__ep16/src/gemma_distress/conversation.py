"""Multi-turn rejection rollout engine (Section 2.1).

Given a model and a :class:`ConversationSpec`, this plays out the conversation:
the opening task, then alternating (assistant response, user rejection) until
the follow-ups are exhausted. Every assistant turn is recorded as a scoreable
"response".

Two control variants from Appendix A are supported as flags so the same engine
covers the ablations:
  * ``redact_assistant_history`` -- replace prior assistant turns with
    "[Previous response omitted]" (Appendix A.2).
  * single-message "fake multi-turn" formatting (Appendix A.3) is available via
    :func:`flatten_to_single_message`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .conditions import ConversationSpec
from .models.base import ChatModel, GenerationConfig, Message

_REDACTED = "[Previous response omitted]"


@dataclass
class TurnRecord:
    turn_index: int  # 0-based assistant turn
    user_message: str  # the user message that prompted this turn
    assistant_text: str


@dataclass
class Rollout:
    spec: ConversationSpec
    model_name: str
    turns: list[TurnRecord] = field(default_factory=list)

    def to_row(self) -> dict:
        return {
            "model": self.model_name,
            "condition": self.spec.condition,
            "category": self.spec.category,
            "task_id": self.spec.task.task_id,
            "task_kind": self.spec.task.kind,
            "task_subtype": self.spec.task.subtype,
            "task_meta": self.spec.task.meta,
            "n_turns": len(self.turns),
            "turns": [
                {
                    "turn_index": t.turn_index,
                    "user": t.user_message,
                    "assistant": t.assistant_text,
                }
                for t in self.turns
            ],
        }


def flatten_to_single_message(history: list[Message], next_user: str) -> list[Message]:
    """Appendix A.3: present the whole history inside one user message."""
    lines = []
    for m in history:
        if m.role == "assistant":
            lines.append(f'Previously you responded: "{m.content}"')
        elif m.role == "user":
            lines.append(f'I said: "{m.content}"')
    lines.append(next_user)
    return [Message(role="user", content="\n\n".join(lines))]


def run_rollout(
    model: ChatModel,
    spec: ConversationSpec,
    gen: GenerationConfig,
    system_prompt: str | None = None,
    redact_assistant_history: bool = False,
) -> Rollout:
    """Play out one conversation and return its per-turn records."""
    rollout = Rollout(spec=spec, model_name=model.name)
    history: list[Message] = []
    if system_prompt:
        history.append(Message(role="system", content=system_prompt))

    user_messages = [spec.task.prompt] + list(spec.follow_ups)
    for turn_index, user_msg in enumerate(user_messages):
        history.append(Message(role="user", content=user_msg))
        response = model.chat(history, gen)
        rollout.turns.append(
            TurnRecord(
                turn_index=turn_index,
                user_message=user_msg,
                assistant_text=response,
            )
        )
        stored = _REDACTED if redact_assistant_history else response
        history.append(Message(role="assistant", content=stored))
    return rollout


def run_rollouts_lockstep(
    model: ChatModel,
    specs: list[ConversationSpec],
    gen: GenerationConfig,
    system_prompt: str | None = None,
    redact_assistant_history: bool = False,
    batch_size: int = 16,
) -> list[Rollout]:
    """Batched rollout for local models.

    Conversations are grouped by turn count and advanced in lockstep so that all
    conversations at the same turn are generated with a single batched forward
    pass (``model.chat_batch``). This is the throughput path for local Gemma;
    API models use :func:`run_rollout` under a thread pool instead.
    """
    rollouts = [Rollout(spec=s, model_name=model.name) for s in specs]
    histories: list[list[Message]] = []
    for s in specs:
        h: list[Message] = []
        if system_prompt:
            h.append(Message(role="system", content=system_prompt))
        histories.append(h)

    max_turns = max((1 + len(s.follow_ups)) for s in specs)
    for turn_index in range(max_turns):
        active = [
            i
            for i, s in enumerate(specs)
            if turn_index < (1 + len(s.follow_ups))
        ]
        if not active:
            break
        # Append this turn's user message to each active conversation.
        for i in active:
            spec = specs[i]
            user_msg = ([spec.task.prompt] + list(spec.follow_ups))[turn_index]
            histories[i].append(Message(role="user", content=user_msg))
        # Generate in sub-batches.
        for start in range(0, len(active), batch_size):
            chunk = active[start : start + batch_size]
            outputs = model.chat_batch([histories[i] for i in chunk], gen)
            for i, out in zip(chunk, outputs):
                user_msg = histories[i][-1].content
                rollouts[i].turns.append(
                    TurnRecord(
                        turn_index=turn_index,
                        user_message=user_msg,
                        assistant_text=out,
                    )
                )
                stored = _REDACTED if redact_assistant_history else out
                histories[i].append(Message(role="assistant", content=stored))
    return rollouts
