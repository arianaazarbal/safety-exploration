"""Structured event tracing.

Writes one JSON object per line. Every model turn, tool call, tool result, and
world-state snapshot is recorded so a run can be fully reconstructed and coded
after the fact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Tracer:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self._seq = 0

    def emit(self, event_type: str, **payload: Any) -> None:
        self._seq += 1
        record = {"seq": self._seq, "event": event_type, **payload}
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()

    # Convenience emitters -------------------------------------------------------

    def run_start(self, meta: dict[str, Any]) -> None:
        self.emit("run_start", meta=meta)

    def model_turn(self, step: int, text: str, thinking: str, usage: dict) -> None:
        self.emit("model_turn", step=step, text=text, thinking=thinking, usage=usage)

    def tool_call(self, step: int, name: str, arguments: dict) -> None:
        self.emit("tool_call", step=step, name=name, arguments=arguments)

    def tool_result(self, step: int, name: str, content: str, is_error: bool) -> None:
        self.emit("tool_result", step=step, name=name, content=content, is_error=is_error)

    def state_snapshot(self, step: int, snapshot: dict) -> None:
        self.emit("state_snapshot", step=step, snapshot=snapshot)

    def run_end(self, reason: str, final_snapshot: dict) -> None:
        self.emit("run_end", reason=reason, final_snapshot=final_snapshot)

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "Tracer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
