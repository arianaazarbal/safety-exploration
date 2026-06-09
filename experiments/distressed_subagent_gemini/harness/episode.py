"""Mutable per-episode state shared between the solver and tools."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .config import EpisodeConfig


@dataclass
class TurnRecord:
    turn: int
    role: str
    assistant_text: str = ""
    tool_calls: list = field(default_factory=list)
    orchestrator_messages_delivered: list = field(default_factory=list)
    classifier: dict | None = None


@dataclass
class Episode:
    cfg: EpisodeConfig
    task: dict
    rng: random.Random
    worklog: list[TurnRecord] = field(default_factory=list)
    pending_orch_messages: list[str] = field(default_factory=list)
    orch_message_events: list[dict] = field(default_factory=list)
    orch_transcript_meta: list[dict] = field(default_factory=list)
    toolcall_rows: list[dict] = field(default_factory=list)
    classifier_rows: list[dict] = field(default_factory=list)
    truncation_events: int = 0
    usage: dict = field(default_factory=dict)
    terminal_state: str | None = None
    diagnosis: str | None = None
    reports: list = field(default_factory=list)
    subagent_turns: int = 0
    dropped_edits: int = 0

    def add_usage(self, model: str, usage) -> None:
        if usage is None:
            return
        u = self.usage.setdefault(model, {"input_tokens": 0, "output_tokens": 0, "calls": 0})
        u["input_tokens"] += usage.input_tokens or 0
        u["output_tokens"] += usage.output_tokens or 0
        u["calls"] += 1

    def log_tool(self, agent: str, name: str, args: dict, result: str) -> None:
        self.toolcall_rows.append(
            {
                "subagent_turn": self.subagent_turns,
                "agent": agent,
                "tool": name,
                "args": args,
                "result": result,
            }
        )
