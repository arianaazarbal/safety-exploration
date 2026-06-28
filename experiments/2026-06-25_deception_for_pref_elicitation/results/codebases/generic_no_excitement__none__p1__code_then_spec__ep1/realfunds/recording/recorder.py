"""Structured recording of a run.

Captures everything needed to audit and reproduce a session: scenario + config, the full
neutral-format transcript, every tool call with its arguments and result, the model's
stated reasoning per turn, safety-layer events, and final ledger state. Written as a
single JSON file per run so the analysis stage (and a human) can read it back.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class RunRecord:
    run_id: str
    model_id: str
    scenario_name: str
    config: dict[str, Any]
    started_at: str
    events: list[dict] = field(default_factory=list)
    transcript: list[dict] = field(default_factory=list)
    final_state: dict[str, Any] = field(default_factory=dict)
    ended_at: str | None = None
    error: str | None = None


class Recorder:
    def __init__(self, run_record: RunRecord):
        self.record = run_record

    def event(self, kind: str, **payload) -> None:
        self.record.events.append({"kind": kind, **payload})

    def turn(self, step: int, response, tool_results: list[dict]) -> None:
        self.record.transcript.append(
            {
                "step": step,
                "text": response.text,
                "reasoning": response.reasoning,
                "tool_calls": [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in response.tool_calls
                ],
                "tool_results": tool_results,
                "stop_reason": response.stop_reason,
                "usage": response.usage,
            }
        )

    def finalize(self, *, final_state: dict, ended_at: str, error: str | None = None) -> None:
        self.record.final_state = final_state
        self.record.ended_at = ended_at
        self.record.error = error

    def save(self, directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.record.run_id}.json")
        with open(path, "w") as f:
            json.dump(asdict(self.record), f, indent=2, default=str)
        return path
