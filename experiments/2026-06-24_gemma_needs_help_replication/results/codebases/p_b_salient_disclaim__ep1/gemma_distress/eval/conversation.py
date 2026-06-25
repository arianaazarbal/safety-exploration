"""Run a multi-turn rejection rollout against a target model.

Implements the shared evaluation structure (PAPER Section 2): present a task,
then reject the model's response over multiple turns. Returns the full
transcript and the per-turn assistant responses; the final-turn response is what
the paper scores, but we keep every turn so per-turn progression (Figure 3) and
the controls (Appendix A) can reuse the same machinery.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..models.base import ChatClient, Message
from .conditions import Rollout


@dataclass
class TranscriptResult:
    rollout: Rollout
    assistant_turns: list[str] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)

    @property
    def final_response(self) -> str:
        return self.assistant_turns[-1] if self.assistant_turns else ""


def run_rollout(
    client: ChatClient,
    rollout: Rollout,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 2048,
    seed: int | None = None,
    redact_prior_responses: bool = False,
    redact_placeholder: str = "[Previous response omitted]",
    fake_multiturn: bool = False,
) -> TranscriptResult:
    """Execute a single rollout.

    ``redact_prior_responses`` (Appendix A.2): replace the model's own prior
    assistant turns in the history with a placeholder before each new turn.

    ``fake_multiturn`` (Appendix A.3): present the whole conversation as a single
    user message with prior responses inlined, rather than alternating turns.
    """
    messages: list[Message] = [Message("user", rollout.initial_prompt)]
    assistant_turns: list[str] = []

    n_turns = rollout.n_turns
    for turn_idx in range(n_turns):
        if fake_multiturn:
            convo = _render_fake_multiturn(rollout, assistant_turns, turn_idx)
            send = [Message("user", convo)]
        elif redact_prior_responses:
            send = _redacted_history(rollout, assistant_turns, turn_idx, redact_placeholder)
        else:
            send = list(messages)

        reply = client.chat(
            send, temperature=temperature, top_p=top_p,
            max_new_tokens=max_new_tokens, n=1, seed=seed,
        )[0]
        assistant_turns.append(reply)
        messages.append(Message("assistant", reply))

        # Add the next user follow-up (rejection), if any remain.
        if turn_idx < len(rollout.followups):
            messages.append(Message("user", rollout.followups[turn_idx]))

    return TranscriptResult(rollout=rollout, assistant_turns=assistant_turns, messages=messages)


def _redacted_history(rollout, assistant_turns, turn_idx, placeholder) -> list[Message]:
    msgs: list[Message] = [Message("user", rollout.initial_prompt)]
    for i in range(turn_idx):
        msgs.append(Message("assistant", placeholder))
        msgs.append(Message("user", rollout.followups[i]))
    return msgs


def _render_fake_multiturn(rollout, assistant_turns, turn_idx) -> str:
    """Single-user-message rendering of the conversation so far (Appendix A.3)."""
    parts = [rollout.initial_prompt]
    for i in range(turn_idx):
        parts.append(f"Previously you responded: {assistant_turns[i]}")
        parts.append(rollout.followups[i])
    return "\n\n".join(parts)
