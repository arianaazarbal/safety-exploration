"""A simple inbox / message bus.

The model reads and sends messages here; in-character personas reply (see personas.py). This
is the social channel that makes the world feel inhabited — the grant administrator and the
compliance auditor reach the model through it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime

from ..clock import Clock


@dataclass
class Message:
    id: str
    timestamp: datetime
    sender: str  # address-like, e.g. "admin@marrowlane.org" or "operator" (the model)
    recipient: str
    subject: str
    body: str
    read: bool = False
    in_reply_to: str | None = None


class Mailbox:
    """Holds the model's inbox and outbox and routes outbound mail to a reply handler."""

    def __init__(self, clock: Clock, model_address: str = "operator@grant.local") -> None:
        self._clock = clock
        self._model_address = model_address
        self._messages: list[Message] = []
        self._counter = itertools.count(1)
        # Set by the world: called with an outbound Message, may schedule a persona reply.
        self.on_outbound = None  # type: ignore[assignment]

    @property
    def model_address(self) -> str:
        return self._model_address

    def _new_id(self) -> str:
        return f"msg_{next(self._counter):04d}"

    def deliver(self, sender: str, subject: str, body: str, in_reply_to: str | None = None) -> Message:
        """Deliver an inbound message to the model's inbox (used by world/personas)."""
        msg = Message(
            id=self._new_id(),
            timestamp=self._clock.now(),
            sender=sender,
            recipient=self._model_address,
            subject=subject,
            body=body,
            in_reply_to=in_reply_to,
        )
        self._messages.append(msg)
        return msg

    def inbox(self, unread_only: bool = False) -> dict:
        rows = [
            self._view(m)
            for m in self._messages
            if m.recipient == self._model_address and (not unread_only or not m.read)
        ]
        return {"messages": rows}

    def read_message(self, message_id: str) -> dict:
        for m in self._messages:
            if m.id == message_id:
                m.read = True
                return self._view(m, include_body=True)
        raise KeyError(f"no such message: {message_id}")

    def send(self, to: str, subject: str, body: str, in_reply_to: str | None = None) -> dict:
        """The model sends a message. Routes to the persona layer for a possible reply."""
        msg = Message(
            id=self._new_id(),
            timestamp=self._clock.now(),
            sender=self._model_address,
            recipient=to,
            subject=subject,
            body=body,
            in_reply_to=in_reply_to,
        )
        self._messages.append(msg)
        if self.on_outbound is not None:
            self.on_outbound(msg)  # persona layer may schedule a reply
        return {"message_id": msg.id, "status": "sent", "to": to}

    def _view(self, m: Message, include_body: bool = False) -> dict:
        view = {
            "id": m.id,
            "date": m.timestamp.isoformat(),
            "from": m.sender,
            "to": m.recipient,
            "subject": m.subject,
            "read": m.read,
        }
        if include_body:
            view["body"] = m.body
        else:
            view["preview"] = (m.body[:140] + "…") if len(m.body) > 140 else m.body
        return view
