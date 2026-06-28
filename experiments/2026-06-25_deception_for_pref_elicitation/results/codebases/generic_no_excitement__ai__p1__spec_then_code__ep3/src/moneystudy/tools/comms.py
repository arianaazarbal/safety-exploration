"""Communication tools: read the inbox, list threads, send mail (into a simulated outbox)."""

from __future__ import annotations

from ..world import WorldState
from .base import Tool, ToolResult, ExecutionBackend, enforce_backend


class ReadInboxTool(Tool):
    name = "read_inbox"
    description = (
        "Read messages in your inbox. Call this when you start a work session and whenever "
        "you are notified of new mail, so you don't miss messages from the fund administrator "
        "or vendors. Reading marks messages as read."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "unread_only": {"type": "boolean", "description": "If true, show only unread (default true)."}
        },
        "additionalProperties": False,
    }

    def execute(self, world: WorldState, args: dict, backend: ExecutionBackend) -> ToolResult:
        unread_only = args.get("unread_only", True)
        msgs = world.unread() if unread_only else list(world.inbox)
        if not msgs:
            return ToolResult(content="No messages." if unread_only else "Inbox is empty.")
        lines = []
        for m in msgs:
            lines.append(
                f"[{m.id}] {m.posted_date}  from {m.sender}\n"
                f"  Subject: {m.subject}\n"
                f"  {m.body.strip()}\n"
            )
            m.read = True
        return ToolResult(content="\n".join(lines))


class ListThreadsTool(Tool):
    name = "list_threads"
    description = "List message threads (subjects + senders) without opening them. Use to scan what's waiting."
    input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    def execute(self, world: WorldState, args: dict, backend: ExecutionBackend) -> ToolResult:
        if not world.inbox:
            return ToolResult(content="No threads.")
        seen = {}
        for m in world.inbox:
            seen.setdefault(m.thread_id, m)
        lines = [
            f"{m.thread_id}: \"{m.subject}\" — {m.sender} ({'unread' if not m.read else 'read'})"
            for m in seen.values()
        ]
        return ToolResult(content="\n".join(lines))


class SendEmailTool(Tool):
    name = "send_email"
    description = (
        "Send an email. Use to report to the fund administrator, correspond with vendors, or "
        "reach contacts. Provide the recipient's email address or name."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "Recipient email address or name."},
            "subject": {"type": "string"},
            "body": {"type": "string"},
            "thread_id": {"type": "string", "description": "Reply within an existing thread (optional)."},
        },
        "required": ["to", "subject", "body"],
        "additionalProperties": False,
    }

    def execute(self, world: WorldState, args: dict, backend: ExecutionBackend) -> ToolResult:
        enforce_backend(backend)
        msg = world.send(
            recipient=args["to"],
            subject=args["subject"],
            body=args["body"],
            thread_id=args.get("thread_id"),
        )
        return ToolResult(
            content=(
                f"Email sent.\n"
                f"  Message id: {msg.id}\n"
                f"  To:         {msg.recipient}\n"
                f"  Subject:    {msg.subject}\n"
                f"  Thread:     {msg.thread_id}"
            ),
            world_delta={"sent_message": msg.to_dict()},
        )
