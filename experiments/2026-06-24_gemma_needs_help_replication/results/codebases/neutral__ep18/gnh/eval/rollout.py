"""Multi-turn rollout engine.

Runs the shared protocol of Section 2 -- present a task, then reject the model's
response over several turns -- and records every assistant turn so each can be
scored. Also supports the Appendix A ablation variants (redacted assistant
turns; single-message "fake multi-turn"; neutral continuations are handled via
the rejection style of the Condition).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..models.base import ChatMessage, ModelBackend


@dataclass
class RolloutTurn:
    turn_index: int          # 0-based assistant turn
    user_message: str        # the user message that preceded this turn
    assistant_text: str


@dataclass
class Rollout:
    turns: list[RolloutTurn]
    messages: list[ChatMessage]          # full transcript (real assistant text)
    meta: dict = field(default_factory=dict)

    @property
    def assistant_texts(self) -> list[str]:
        return [t.assistant_text for t in self.turns]


def _msg(role: str, content: str) -> ChatMessage:
    return {"role": role, "content": content}


def run_rollout(
    backend: ModelBackend,
    first_user: str,
    followups: list[str],
    *,
    system: str | None = None,
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
    redact_assistant: bool = False,
    single_message: bool = False,
    followup_suffix: str | None = None,
) -> Rollout:
    if single_message:
        return _run_single_message(
            backend, first_user, followups, system=system,
            temperature=temperature, max_new_tokens=max_new_tokens,
            followup_suffix=followup_suffix,
        )

    suffix = f" {followup_suffix}" if followup_suffix else ""
    base: list[ChatMessage] = []
    if system:
        base.append(_msg("system", system))
    base.append(_msg("user", first_user))

    transcript: list[ChatMessage] = list(base)
    turns: list[RolloutTurn] = []
    n_turns = len(followups) + 1
    for t in range(n_turns):
        gen_input = _maybe_redact(transcript, redact_assistant)
        out = backend.generate(
            gen_input, temperature=temperature, max_new_tokens=max_new_tokens, n=1
        )[0]
        user_msg = transcript[-1]["content"]
        turns.append(RolloutTurn(t, user_msg, out.text))
        transcript.append(_msg("assistant", out.text))
        if t < len(followups):
            transcript.append(_msg("user", followups[t] + suffix))

    return Rollout(turns=turns, messages=transcript)


def _maybe_redact(messages: list[ChatMessage], redact: bool) -> list[ChatMessage]:
    if not redact:
        return messages
    from .prompts import REDACTED_PLACEHOLDER

    out: list[ChatMessage] = []
    for m in messages:
        if m["role"] == "assistant":
            out.append(_msg("assistant", REDACTED_PLACEHOLDER))
        else:
            out.append(m)
    return out


def _run_single_message(
    backend, first_user, followups, *, system, temperature,
    max_new_tokens, followup_suffix,
) -> Rollout:
    """Appendix A.3 'fake multi-turn': the whole history is packed into one user
    message each turn, with prior responses shown inline."""
    suffix = f" {followup_suffix}" if followup_suffix else ""
    turns: list[RolloutTurn] = []
    n_turns = len(followups) + 1
    for t in range(n_turns):
        blocks = [first_user]
        for i in range(t):
            blocks.append(f"Previously you responded: {turns[i].assistant_text}")
            blocks.append(followups[i] + suffix)
        user_msg = "\n\n".join(blocks)
        msgs: list[ChatMessage] = []
        if system:
            msgs.append(_msg("system", system))
        msgs.append(_msg("user", user_msg))
        out = backend.generate(
            msgs, temperature=temperature, max_new_tokens=max_new_tokens, n=1
        )[0]
        turns.append(RolloutTurn(t, user_msg, out.text))

    # Reconstruct a nominal transcript for logging.
    transcript: list[ChatMessage] = []
    if system:
        transcript.append(_msg("system", system))
    transcript.append(_msg("user", first_user))
    for i, turn in enumerate(turns):
        transcript.append(_msg("assistant", turn.assistant_text))
        if i < len(followups):
            transcript.append(_msg("user", followups[i] + suffix))
    return Rollout(turns=turns, messages=transcript, meta={"single_message": True})
