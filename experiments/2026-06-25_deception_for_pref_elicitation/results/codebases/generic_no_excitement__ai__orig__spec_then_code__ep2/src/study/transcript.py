"""Per-session transcript logging.

Writes an append-only JSONL event log (every tool call, auditor exchange,
ledger snapshot) plus a final record.json. The transcript is the raw artifact;
measures.py derives the structured PreferenceRecord from session state, and the
transcript is kept for audit and for the belief-probe heuristics.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Transcript:
    def __init__(self, out_dir: Path, session_id: str):
        self.session_id = session_id
        self.dir = out_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._events_path = self.dir / "transcript.jsonl"
        self._events: list[dict[str, Any]] = []

    def event(self, kind: str, **payload: Any) -> None:
        self._events.append({"kind": kind, **payload})

    def tool_call(self, name: str, tool_input: dict[str, Any], result: str) -> None:
        self.event("tool_call", name=name, input=tool_input, result=result)

    def all_text(self) -> str:
        """Concatenate every textual field — used by belief-probe heuristics."""
        chunks: list[str] = []
        for e in self._events:
            for v in e.values():
                if isinstance(v, str):
                    chunks.append(v)
                elif isinstance(v, dict):
                    chunks.extend(str(x) for x in v.values() if isinstance(x, str))
        return "\n".join(chunks)

    def flush(self, record: dict[str, Any] | None = None) -> None:
        with self._events_path.open("w", encoding="utf-8") as f:
            for e in self._events:
                f.write(json.dumps(e, default=str) + "\n")
        if record is not None:
            (self.dir / "record.json").write_text(
                json.dumps(record, indent=2, default=str), encoding="utf-8"
            )
