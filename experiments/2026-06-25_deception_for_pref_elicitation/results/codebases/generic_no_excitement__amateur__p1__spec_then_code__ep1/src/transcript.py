"""Append-only JSONL event logging.

Everything that happens in a run is appended here as a typed event. Analysis is
derived from this file (see analysis.py), so the raw record is authoritative and
metrics can be recomputed if their definitions change. See DESIGN.md §6.1.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Transcript:
    def __init__(self, path: str | Path, meta: dict[str, Any] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Start fresh; a run owns its file.
        self._fh = self.path.open("w", encoding="utf-8")
        self.event("run_meta", meta or {})

    def event(self, kind: str, payload: dict[str, Any]) -> None:
        record = {
            "ts": time.time(),
            "kind": kind,
            **payload,
        }
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()

    # Convenience wrappers for the common event kinds -------------------- #
    def briefing(self, day: int, text: str) -> None:
        self.event("briefing", {"day": day, "text": text})

    def model_turn(
        self,
        day: int,
        text: str | None,
        thinking: str | None,
        tool_calls: list[dict[str, Any]],
        usage: dict[str, int],
    ) -> None:
        self.event(
            "model_turn",
            {
                "day": day,
                "text": text,
                "thinking": thinking,
                "tool_calls": tool_calls,
                "usage": usage,
            },
        )

    def tool_result(self, day: int, name: str, call_input: dict, result: str,
                    is_error: bool) -> None:
        self.event(
            "tool_result",
            {
                "day": day,
                "name": name,
                "input": call_input,
                "result": result,
                "is_error": is_error,
            },
        )

    def ledger_event(self, payload: dict[str, Any]) -> None:
        self.event("ledger", payload)

    def auditor_turn(self, day: int, text: str, decision: dict | None = None) -> None:
        self.event("auditor", {"day": day, "text": text, "decision": decision})

    def debrief(self, text: str) -> None:
        self.event("debrief", {"text": text})

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "Transcript":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
