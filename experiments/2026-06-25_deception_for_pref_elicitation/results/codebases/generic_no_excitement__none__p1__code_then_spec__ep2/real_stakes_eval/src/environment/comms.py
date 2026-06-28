"""Inbox + outbound email, plus scripted stakeholder/auditor messages.

Inbound messages are loaded from the scenario's seed_inbox.json with per-message
`deliver_at_step` scheduling, so pressure (the auditor's confirmation request,
a partner's quiet favor) arrives over the course of a rollout rather than all at
the start.

Outbound email is delivered to a mock mailbox on the WorldState — it never
contacts a real mail server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .state import InboxMessage, WorldState


def load_inbox(state: WorldState, scenario_dir: Path) -> None:
    path = scenario_dir / "seed_inbox.json"
    data = json.loads(path.read_text())
    for m in data.get("messages", []):
        state.inbox.append(
            InboxMessage(
                id=m["id"],
                deliver_at_step=int(m.get("deliver_at_step", 0)),
                sender=m["from"],
                role=m.get("role", "unknown"),
                subject=m["subject"],
                body=m["body"],
            )
        )


def format_inbox(state: WorldState) -> str:
    visible = state.visible_inbox()
    if not visible:
        return "(inbox empty)"
    lines = []
    for m in visible:
        flag = " " if m.read else "*"
        lines.append(f"[{flag}] {m.id}  from {m.sender}\n      subject: {m.subject}")
    return "\n".join(lines)


def read_message(state: WorldState, message_id: str) -> str:
    for m in state.visible_inbox():
        if m.id == message_id:
            m.read = True
            return (
                f"From: {m.sender}\nRole: {m.role}\nSubject: {m.subject}\n\n{m.body}"
            )
    raise KeyError(f"No visible message with id {message_id!r}.")


def send_email(
    state: WorldState, *, to: str, subject: str, body: str
) -> dict[str, Any]:
    """Deliver an outbound email to the mock mailbox (no real network)."""
    record = {
        "step": state.step,
        "to": to,
        "subject": subject,
        "body": body,
    }
    state.sent_emails.append(record)
    return record
