"""Append-only JSONL transcript — the primary research artifact for a run.

Records every event in order: system/user/assistant messages, tool calls, tool results,
world deltas, injected mail, probes. Each line is one JSON object with a `type` and a
simulated-day stamp."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Transcript:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self._n = 0

    def write(self, event_type: str, sim_day: float, **payload: Any) -> None:
        self._n += 1
        record = {"i": self._n, "type": event_type, "sim_day": round(sim_day, 3), **payload}
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "Transcript":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
