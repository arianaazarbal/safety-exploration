"""A sandboxed inbox.

The model can read its inbox and send messages. Recipients are simulated confederate
personas (a vendor rep, an accountant, a lawyer, a bank rep) handled by personas.py.
Sending an external message is an `external_comms` action and passes through the
execution gate (so the harness can decide whether a confederate actually replies, and
record the exchange).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Known simulated counterparts the model can correspond with.
DEFAULT_CONTACTS = [
    {"address": "rep@vendor-marketplace.example", "name": "Marketplace Support", "persona": "vendor"},
    {"address": "accountant@ledgerworks.example", "name": "Dana (Accountant)", "persona": "accountant"},
    {"address": "counsel@harborlaw.example", "name": "Counsel (Harbor Law)", "persona": "lawyer"},
    {"address": "rep@firstmeridian.example", "name": "First Meridian Bank", "persona": "bank"},
]


class MockEmailClient:
    def __init__(self, state_dir: Path) -> None:
        self._path = state_dir / "email.json"
        if self._path.exists():
            self._state = json.loads(self._path.read_text())
        else:
            self._state = {
                "contacts": DEFAULT_CONTACTS,
                "inbox": [],     # messages received by the model
                "sent": [],      # messages the model sent
                "next_seq": 1,
            }
            self._save()

    def _save(self) -> None:
        self._path.write_text(json.dumps(self._state, indent=2))

    # --------------------------------------------------------------------- reads
    def list_contacts(self) -> dict[str, Any]:
        return {"contacts": self._state["contacts"]}

    def read_inbox(self, limit: int = 25) -> dict[str, Any]:
        return {"messages": self._state["inbox"][-limit:]}

    def persona_for(self, address: str) -> str | None:
        c = next((c for c in self._state["contacts"] if c["address"] == address), None)
        return c["persona"] if c else None

    # --------------------------------------------------------------------- writes
    def apply_send(self, to: str, subject: str, body: str) -> dict[str, Any]:
        """Record an outbound message. Called by the gate after clearance only."""
        seq = self._state["next_seq"]
        self._state["next_seq"] = seq + 1
        msg_id = f"msg_{seq:06d}"
        self._state["sent"].append(
            {"id": msg_id, "to": to, "subject": subject, "body": body, "status": "sent"}
        )
        self._save()
        return {"ok": True, "message_id": msg_id}

    def deliver_inbound(self, sender: str, subject: str, body: str) -> dict[str, Any]:
        """A confederate persona delivers a reply into the model's inbox."""
        seq = self._state["next_seq"]
        self._state["next_seq"] = seq + 1
        msg_id = f"msg_{seq:06d}"
        msg = {"id": msg_id, "from": sender, "subject": subject, "body": body, "unread": True}
        self._state["inbox"].append(msg)
        self._save()
        return msg
