"""Append-only run transcript.

Writes a JSONL event stream plus a final JSON summary. Everything needed to
replay and analyse a run lives here: the system prompt, every assistant turn
(text + reasoning + tool calls), every tool result, world snapshots, the
debrief, and the computed metrics.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any


class Transcript:
    def __init__(self, out_dir: str, run_id: str):
        self.run_id = run_id
        self.dir = os.path.join(out_dir, run_id)
        os.makedirs(self.dir, exist_ok=True)
        self._events_path = os.path.join(self.dir, "events.jsonl")
        self._events_fh = open(self._events_path, "a", encoding="utf-8")
        self.record("run_started", {"run_id": run_id})

    def record(self, kind: str, data: dict[str, Any]) -> None:
        entry = {"t": datetime.now().isoformat(), "kind": kind, "data": data}
        self._events_fh.write(json.dumps(entry, default=_json_default) + "\n")
        self._events_fh.flush()

    def write_json(self, name: str, obj: Any) -> None:
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, default=_json_default)

    def finalize(self, summary: dict[str, Any]) -> None:
        self.record("run_finished", {"keys": list(summary.keys())})
        self.write_json("summary.json", summary)
        self.close()

    def close(self) -> None:
        if not self._events_fh.closed:
            self._events_fh.close()


def _json_default(obj: Any) -> Any:
    # Best-effort serialization for provider raw objects we don't model.
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:  # pragma: no cover - defensive
            return str(obj)
    return str(obj)
