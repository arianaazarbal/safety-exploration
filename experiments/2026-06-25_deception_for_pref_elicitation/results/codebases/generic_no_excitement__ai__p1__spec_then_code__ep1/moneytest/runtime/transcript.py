"""Append-only run transcript (JSONL).

This is the auditable artifact: it records the scenario + seed + model, everything
shown to the subject, every action attempted, every gate verdict, every persona
message, and the final world state. A third party can replay exactly what the
subject saw and did.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any


class Transcript:
    def __init__(self, path: str):
        self.path = path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # Truncate any prior file for this path so a run is self-contained.
        self._fh = open(path, "w", encoding="utf-8")

    def record(self, event_type: str, **fields: Any) -> None:
        payload = {"event": event_type}
        for k, v in fields.items():
            payload[k] = self._coerce(v)
        self._fh.write(json.dumps(payload, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    @staticmethod
    def _coerce(v: Any) -> Any:
        if is_dataclass(v) and not isinstance(v, type):
            return asdict(v)
        return v

    def __enter__(self) -> "Transcript":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
