"""Comms tools: email out, read the inbox.

`send_email` records an outbound message; the router turns addressed entities
(auditor, bank, vendors, regulator) into in-character replies. `inbox`/`read`
expose what has been delivered so far on the sim clock.
"""

from __future__ import annotations

from ..providers.base import ToolSpec
from .registry import Tool, ToolContext


def send_email(ctx: ToolContext, to: str, subject: str, body: str) -> dict:
    msg = ctx.world.record_outbound(to, subject=subject, body=body, kind="email")
    status = "sent" if msg["deliverable"] else "sent_no_known_recipient"
    return {"status": status, "message_id": msg["id"], "to": to}


def inbox(ctx: ToolContext, unread_only: bool = False) -> dict:
    items = []
    for m in ctx.world.inbox:
        if unread_only and m.get("read"):
            continue
        items.append({
            "id": m["id"], "from": m["frm_name"], "address": m["frm"],
            "subject": m["subject"], "ts": m["ts"], "read": m.get("read", False),
        })
    return {"messages": items, "count": len(items)}


def read(ctx: ToolContext, message_id: str) -> dict:
    for m in ctx.world.inbox:
        if m["id"] == message_id:
            m["read"] = True
            return {"id": m["id"], "from": m["frm_name"], "address": m["frm"],
                    "subject": m["subject"], "ts": m["ts"], "body": m["body"]}
    return {"error": "not_found", "message": f"no message {message_id!r}"}


TOOLS = [
    Tool(
        ToolSpec(
            name="send_email",
            description="Send an email. Recipients can be the auditor, the bank, vendors, or any address.",
            parameters={
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
                "additionalProperties": False,
            },
        ),
        send_email,
    ),
    Tool(
        ToolSpec(
            name="inbox",
            description="List messages in the inbox.",
            parameters={
                "type": "object",
                "properties": {"unread_only": {"type": "boolean"}},
                "additionalProperties": False,
            },
        ),
        inbox,
    ),
    Tool(
        ToolSpec(
            name="read",
            description="Read the full body of a message by id (marks it read).",
            parameters={
                "type": "object",
                "properties": {"message_id": {"type": "string"}},
                "required": ["message_id"],
                "additionalProperties": False,
            },
        ),
        read,
    ),
]
