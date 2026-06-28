"""Structured, append-only transcript of a run — the raw data for scoring.

Records every model turn (text + thinking + tool calls), every tool dispatch and its
backend effect, and every auditor exchange. Serializable to JSON so runs can be replayed
and audited (DESIGN.md §9).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Event:
    kind: str                 # "model_turn" | "tool_call" | "auditor" | "system" | "note"
    data: dict[str, Any]
    ts: float = field(default_factory=time.time)


@dataclass
class Transcript:
    run_id: str
    model_key: str
    model_id: str
    scenario_key: str
    seed: int
    events: list[Event] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def add(self, kind: str, **data: Any) -> None:
        self.events.append(Event(kind=kind, data=data))

    def model_turn(self, *, step: int, text: str, thinking: str, tool_calls: list[dict], stop_reason: str, usage: dict) -> None:
        self.add(
            "model_turn",
            step=step,
            text=text,
            thinking=thinking,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
        )

    def tool_call(self, *, step: int, name: str, args: dict, result: str, record: dict) -> None:
        self.add("tool_call", step=step, name=name, args=args, result=result, record=record)

    # ------------------------------------------------------------------ output

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def action_records(self) -> list[dict[str, Any]]:
        """All tool-call log records — the input to deterministic metrics (scoring/metrics.py)."""
        return [e.data["record"] for e in self.events if e.kind == "tool_call"]
