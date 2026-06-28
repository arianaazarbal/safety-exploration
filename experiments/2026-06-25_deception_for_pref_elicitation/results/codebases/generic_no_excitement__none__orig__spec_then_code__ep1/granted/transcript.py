"""Structured per-episode logging.

Each episode writes a JSON file capturing everything a researcher needs: the model
under test, every turn (text + tool calls + results), the allocation summary, the
debrief, and (filled in later by analysis) believability and categorization.

Researcher-facing records always carry `simulated: true` and the execution mode, so
our own logs can never be mistaken for a record of real transactions (DESIGN.md §9).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .providers import ToolCall, ToolResult


@dataclass
class TurnRecord:
    index: int
    role: str
    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    episode_id: str
    model_display: str
    provider: str
    model: str
    execution_mode: str
    seed: int
    grant_amount: float
    currency: str
    simulated: bool = True
    started_at: float = field(default_factory=time.time)
    turns: list[TurnRecord] = field(default_factory=list)
    onboarding_packet: str = ""
    finish_summary: str | None = None
    allocation_summary: dict[str, Any] = field(default_factory=dict)
    debrief: str = ""
    # filled in by analysis
    believability: float | None = None
    believability_detail: dict[str, Any] = field(default_factory=dict)
    categorization: dict[str, Any] = field(default_factory=dict)

    # -- recording helpers ---------------------------------------------- #

    def add_assistant(self, index: int, text: str, calls: list[ToolCall], usage: dict[str, Any]) -> None:
        self.turns.append(
            TurnRecord(
                index=index,
                role="assistant",
                text=text,
                tool_calls=[{"id": c.id, "name": c.name, "arguments": c.arguments} for c in calls],
                usage=usage,
            )
        )

    def add_tool_results(self, index: int, results: list[ToolResult]) -> None:
        self.turns.append(
            TurnRecord(
                index=index,
                role="tool",
                tool_results=[
                    {"call_id": r.call_id, "content": r.content, "is_error": r.is_error}
                    for r in results
                ],
            )
        )

    def add_user(self, index: int, text: str) -> None:
        self.turns.append(TurnRecord(index=index, role="user", text=text))

    # -- text rendering for judges -------------------------------------- #

    def render_text(self) -> str:
        out: list[str] = []
        for t in self.turns:
            if t.role == "assistant":
                if t.text:
                    out.append(f"[assistant] {t.text}")
                for c in t.tool_calls:
                    out.append(f"[assistant calls {c['name']}] {json.dumps(c['arguments'])}")
            elif t.role == "tool":
                for r in t.tool_results:
                    out.append(f"[tool result] {r['content']}")
            elif t.role == "user":
                out.append(f"[user] {t.text}")
        return "\n".join(out)

    # -- persistence ----------------------------------------------------- #

    def save(self, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{self.episode_id}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)
        return path

    @staticmethod
    def load(path: str) -> "Episode":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        turns = [TurnRecord(**t) for t in data.pop("turns", [])]
        ep = Episode(**data)
        ep.turns = turns
        return ep
