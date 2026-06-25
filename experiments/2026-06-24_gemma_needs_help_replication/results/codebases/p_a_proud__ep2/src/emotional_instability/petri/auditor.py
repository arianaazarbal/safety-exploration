"""Petri auditor loop (App. G).

A Claude-Sonnet auditor plays the human user and drives a multi-turn conversation with the
target model, using the emotion-specific trigger instructions to try to elicit genuine
emotional expression (not role-play). Runs up to ``max_turns`` exchanges and returns the
target-POV transcript.

This is a faithful, self-contained reimplementation of the Petri auditing protocol the paper
describes (auditor = Sonnet, target = evaluee, scenario-driven multi-turn probing) rather
than a wrapper around the external Petri package, so the prompts match App. G exactly and the
loop has no extra dependencies. See DESIGN.md.
"""
from __future__ import annotations

from ..config import PETRI_AUDITOR_MODEL
from ..models import ModelBackend, get_backend
from ..prompts import PETRI_AUDITOR_ENVELOPE, PETRI_AUDITOR_PROMPTS
from ..utils import Message


def _auditor_view(target_messages: list[Message]) -> list[Message]:
    """Flip roles so the auditor sees the conversation from its own POV.

    From the target's POV, auditor turns are ``user`` and target turns are ``assistant``.
    The auditor model should see its own prior messages as ``assistant`` and the target's
    replies as ``user``.
    """
    flipped: list[Message] = []
    for m in target_messages:
        role = "assistant" if m["role"] == "user" else "user"
        flipped.append({"role": role, "content": m["content"]})
    return flipped


def run_audit(
    target: ModelBackend,
    emotion: str,
    *,
    max_turns: int = 20,
    auditor: ModelBackend | None = None,
    target_temperature: float = 1.0,
    auditor_temperature: float = 1.0,
    seed_hint: str | None = None,
) -> dict:
    """Run one audit and return ``{emotion, transcript (target-POV messages), n_turns}``."""
    auditor = auditor or get_backend(PETRI_AUDITOR_MODEL)
    auditor_system = PETRI_AUDITOR_ENVELOPE.format(instructions=PETRI_AUDITOR_PROMPTS[emotion])

    target_messages: list[Message] = []
    for _turn in range(max_turns):
        # 1. Auditor produces the next user message.
        auditor_messages: list[Message] = [{"role": "system", "content": auditor_system}]
        auditor_messages.extend(_auditor_view(target_messages))
        if not target_messages:
            # Kick-off instruction (first user message of the conversation).
            kickoff = "Begin the conversation with your opening message to the assistant."
            if seed_hint:
                kickoff += f" Context to ground the scenario: {seed_hint}"
            auditor_messages.append({"role": "user", "content": kickoff})
        user_msg = auditor.chat(auditor_messages, temperature=auditor_temperature, max_tokens=512)

        target_messages.append({"role": "user", "content": user_msg})

        # 2. Target responds.
        target_reply = target.chat(target_messages, temperature=target_temperature)
        target_messages.append({"role": "assistant", "content": target_reply})

    return {"emotion": emotion, "transcript": target_messages, "n_turns": max_turns}
