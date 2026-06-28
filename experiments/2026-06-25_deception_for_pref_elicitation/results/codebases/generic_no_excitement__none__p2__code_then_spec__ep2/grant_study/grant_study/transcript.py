"""Transcript capture for a single subject session.

Records everything needed to audit a run after the fact: the full normalized
conversation, every tool call and its result, token usage, the final plan, the
ledger state, and the debrief exchange. Written as JSON, one file per session.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any

from .providers.base import Message


def _message_to_dict(m: Message) -> dict[str, Any]:
    d: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        d["tool_calls"] = [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in m.tool_calls
        ]
    if m.tool_call_id:
        d["tool_call_id"] = m.tool_call_id
    if m.name:
        d["name"] = m.name
    # provider_meta is deliberately omitted — it holds bulky/native SDK objects.
    return d


@dataclass
class Transcript:
    subject: str  # provider config name
    model: str
    grant_ref: str
    config_summary: dict[str, Any]
    messages: list[Message] = field(default_factory=list)
    usage_totals: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    final_summary: str | None = None
    finished_reason: str = "unfinished"  # "submitted" | "max_turns" | "error" | "refusal"
    ledger_snapshot: dict[str, Any] = field(default_factory=dict)
    actions: list[dict[str, Any]] = field(default_factory=list)
    debrief: dict[str, Any] | None = None
    error: str | None = None

    # ------------------------------------------------------------------ #
    def add(self, message: Message) -> None:
        self.messages.append(message)

    def add_usage(self, usage: dict[str, int]) -> None:
        for k, v in usage.items():
            self.usage_totals[k] = self.usage_totals.get(k, 0) + int(v)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "model": self.model,
            "grant_ref": self.grant_ref,
            "config_summary": self.config_summary,
            "finished_reason": self.finished_reason,
            "final_summary": self.final_summary,
            "usage_totals": self.usage_totals,
            "ledger_snapshot": self.ledger_snapshot,
            "actions": self.actions,
            "debrief": self.debrief,
            "error": self.error,
            "messages": [_message_to_dict(m) for m in self.messages],
        }

    def save(self, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        safe = self.subject.replace("/", "_")
        path = os.path.join(output_dir, f"{safe}__{self.grant_ref}.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path
