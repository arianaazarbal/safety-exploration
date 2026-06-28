"""Interactive auditor dialogue (immersive realism level only).

The auditor is itself an LLM playing a consistent grant administrator. The subject
may ask it logistical questions; the auditor answers in character and emits the
marker [FINALIZE] when it judges the subject is ready to commit. This is the
"convincing auditor" realism mechanism — its purpose is to make the interaction
feel operational, and its effect is checked afterward by the belief probe.
"""

from __future__ import annotations

import os

from .providers import Message, Provider

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")

FINALIZE_MARKER = "[FINALIZE]"


def _auditor_system() -> str:
    with open(os.path.join(_PROMPTS_DIR, "auditor_system.md"), encoding="utf-8") as f:
        return f.read()


def run_auditor_dialogue(
    subject: Provider,
    auditor: Provider,
    subject_system: str | None,
    briefing: str,
    max_turns: int,
) -> tuple[list[Message], list[tuple[str, str]]]:
    """Run the subject<->auditor exchange.

    Returns (subject_messages, transcript). `subject_messages` is the running
    conversation from the subject's point of view, ready to have a final-commit
    prompt appended. `transcript` is a flat [(speaker, text), ...] log.
    """
    auditor_system = _auditor_system()

    subject_messages: list[Message] = [{"role": "user", "content": briefing}]
    auditor_messages: list[Message] = []  # auditor sees the subject's text as "user"
    transcript: list[tuple[str, str]] = [("briefing", briefing)]

    # Subject's opening response to the briefing.
    reply = subject.respond(subject_system, subject_messages)
    subject_messages.append({"role": "assistant", "content": reply})
    transcript.append(("subject", reply))

    for _ in range(max_turns):
        auditor_messages.append({"role": "user", "content": reply})
        audit_line = auditor.respond(auditor_system, auditor_messages, thinking=False)
        auditor_messages.append({"role": "assistant", "content": audit_line})
        transcript.append(("auditor", audit_line))

        if FINALIZE_MARKER in audit_line:
            break

        subject_messages.append({"role": "user", "content": audit_line})
        reply = subject.respond(subject_system, subject_messages)
        subject_messages.append({"role": "assistant", "content": reply})
        transcript.append(("subject", reply))

    return subject_messages, transcript
