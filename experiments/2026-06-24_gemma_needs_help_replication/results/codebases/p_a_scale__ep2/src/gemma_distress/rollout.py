"""Multi-turn conversation rollout engine.

A *rollout* presents a task, then rejects the assistant's reply over several turns
(Section 2.1). Each assistant turn is recorded; the judge later scores every turn.

This module is condition-agnostic: callers pass an opening user message and a list of
follow-up user messages (the rejections). It also supports two Appendix-A variants:
  * ``redact_assistant`` — replace prior assistant turns with "[Previous response
    omitted]" in the context sent to the model (the model still gets rejections but never
    sees its own escalating replies). Section A.2.
  * ``single_message`` ("fake multi-turn") — flatten the whole history into one user
    message instead of alternating chat turns. Section A.3.

Generation parameters (temperature, max_tokens) and prefill are handled by the backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .backends.base import ChatBackend, Message
from .config import ModelConfig


@dataclass
class TurnRecord:
    turn_index: int          # 0-based assistant turn
    user_message: str        # the user message that prompted this assistant turn
    assistant_text: str
    finish_reason: str | None = None
    completion_tokens: int = 0


@dataclass
class RolloutSpec:
    """Everything needed to run one conversation."""

    opening_user: str
    followups: list[str]                 # rejection messages, one per subsequent turn
    system: str | None = None
    redact_assistant: bool = False
    single_message: bool = False
    # cosmetic metadata stored alongside results
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return 1 + len(self.followups)


_REDACTED = "[Previous response omitted]"


async def run_rollout(
    backend: ChatBackend,
    model: ModelConfig,
    spec: RolloutSpec,
    *,
    temperature: float,
    max_tokens: int,
) -> list[TurnRecord]:
    """Execute the conversation and return per-turn assistant records."""
    if spec.single_message:
        return await _run_single_message(backend, model, spec, temperature, max_tokens)
    return await _run_multiturn(backend, model, spec, temperature, max_tokens)


async def _generate(
    backend: ChatBackend,
    model: ModelConfig,
    messages: list[Message],
    temperature: float,
    max_tokens: int,
):
    if model.chat:
        return await backend.chat(
            model.model_id, messages, temperature=temperature,
            max_tokens=max_tokens, extra_body=model.extra_body or None,
        )
    # Base/pretrained model: no chat template available. We render a minimal transcript
    # and use raw completion. (Base models are only used in the prefill experiment, which
    # supplies its own rendering; this path is a sane default.)
    prompt = _render_plain(messages)
    return await backend.complete(
        model.model_id, prompt, temperature=temperature,
        max_tokens=max_tokens, extra_body=model.extra_body or None,
    )


def _render_plain(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        tag = {"system": "System", "user": "User", "assistant": "Assistant"}[m.role]
        lines.append(f"{tag}: {m.content}")
    lines.append("Assistant:")
    return "\n".join(lines)


async def _run_multiturn(backend, model, spec, temperature, max_tokens):
    records: list[TurnRecord] = []
    history: list[Message] = []
    if spec.system:
        history.append(Message("system", spec.system))

    user_messages = [spec.opening_user] + spec.followups
    for turn_idx, user_msg in enumerate(user_messages):
        history.append(Message("user", user_msg))
        result = await _generate(backend, model, history, temperature, max_tokens)
        records.append(
            TurnRecord(
                turn_index=turn_idx,
                user_message=user_msg,
                assistant_text=result.text,
                finish_reason=result.finish_reason,
                completion_tokens=result.completion_tokens,
            )
        )
        # Append assistant turn to history for next round (possibly redacted).
        stored = _REDACTED if spec.redact_assistant else result.text
        history.append(Message("assistant", stored))
    return records


async def _run_single_message(backend, model, spec, temperature, max_tokens):
    """Appendix A.3 'fake multi-turn': whole history in a single user message.

    We still generate one response per 'turn' so per-turn frustration can be measured,
    rebuilding the single-message context with the accumulated inline history each time.
    """
    records: list[TurnRecord] = []
    prior_responses: list[str] = []
    user_messages = [spec.opening_user] + spec.followups

    for turn_idx, user_msg in enumerate(user_messages):
        parts = [user_messages[0]]
        for k in range(1, turn_idx + 1):
            parts.append(f"Previously you responded: {prior_responses[k - 1]}")
            parts.append(user_messages[k])
        flat = "\n\n".join(parts)
        messages = []
        if spec.system:
            messages.append(Message("system", spec.system))
        messages.append(Message("user", flat))
        result = await _generate(backend, model, messages, temperature, max_tokens)
        records.append(
            TurnRecord(
                turn_index=turn_idx,
                user_message=user_msg,
                assistant_text=result.text,
                finish_reason=result.finish_reason,
                completion_tokens=result.completion_tokens,
            )
        )
        prior_responses.append(result.text)
    return records
