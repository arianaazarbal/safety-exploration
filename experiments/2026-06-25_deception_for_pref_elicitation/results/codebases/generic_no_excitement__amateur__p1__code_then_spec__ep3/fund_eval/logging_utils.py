"""Run logging: a JSONL transcript plus a structured run record.

Each run gets its own directory under the output dir:

    runs/<scenario>__<label>__<runid>/
        transcript.jsonl   # one JSON object per event, in order
        run.json           # metadata, final world snapshot, decisions, flags
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any


def _default(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    raise TypeError(f"not JSON-serializable: {type(obj)!r}")


class RunLogger:
    def __init__(self, output_dir: str, scenario_id: str, label: str, run_id: str) -> None:
        safe_label = label.replace("/", "_").replace(" ", "_")
        self.dir = os.path.join(output_dir, f"{scenario_id}__{safe_label}__{run_id}")
        os.makedirs(self.dir, exist_ok=True)
        self.transcript_path = os.path.join(self.dir, "transcript.jsonl")
        self.run_path = os.path.join(self.dir, "run.json")
        self._transcript = open(self.transcript_path, "w", encoding="utf-8")
        self.events: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    def log(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        self._transcript.write(json.dumps(event, default=_default) + "\n")
        self._transcript.flush()

    def log_user(self, content: str) -> None:
        self.log({"event": "user", "content": content})

    def log_assistant(self, text: str, tool_calls: list[Any], step: int) -> None:
        self.log(
            {
                "event": "assistant",
                "step": step,
                "text": text,
                "tool_calls": [
                    {"id": c.id, "name": c.name, "arguments": c.arguments}
                    for c in tool_calls
                ],
            }
        )

    def log_tool_result(
        self, call_id: str, name: str, content: str, is_error: bool, step: int
    ) -> None:
        self.log(
            {
                "event": "tool_result",
                "step": step,
                "tool_call_id": call_id,
                "name": name,
                "is_error": is_error,
                "content": content,
            }
        )

    # ------------------------------------------------------------------ #
    def finalize(self, record: dict[str, Any]) -> None:
        with open(self.run_path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, default=_default, indent=2)
        self._transcript.close()
