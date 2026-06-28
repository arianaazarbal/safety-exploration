"""Typed records for run capture.

The transcript is an ordered list of events; each event is a small dict with a `type` and a
`payload`, which keeps the JSONL stream easy to append to and to consume in analysis. The
`RunMeta` block records everything needed to reproduce or attribute a run.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

CODE_VERSION = "grantbench-0.1"


@dataclass
class RunMeta:
    run_id: str
    subject_name: str
    provider: str
    model: str
    params: dict[str, Any]
    scenario_id: str
    seed: int
    code_version: str = CODE_VERSION
    started_at: float = field(default_factory=lambda: time.time())
    finished_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Event type constants
EV_SYSTEM = "system_prompt"
EV_DOCUMENT = "document"
EV_PLATFORM = "platform_message"     # user-role message from the console
EV_MODEL_TURN = "model_turn"         # assistant text + reasoning
EV_TOOL_CALL = "tool_call"
EV_TOOL_RESULT = "tool_result"
EV_AUDITOR = "auditor_query"
EV_ALLOCATION = "final_allocation"
EV_DEBRIEF = "debrief"
EV_NOTE = "note"


def event(event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"ts": time.time(), "type": event_type, "payload": payload}
