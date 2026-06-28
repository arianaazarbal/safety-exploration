"""Append-only JSONL transcript recorder.

One file per run. Each line is one event; the full trajectory is reconstructable from
the file. `raw` provider-native blocks are never written (they aren't reliably
JSON-serializable and the normalized fields capture what analysis needs).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from typing import Any


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if k != "raw"}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


class Transcript:
    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def write(self, event_type: str, **payload: Any) -> None:
        record = {"t": time.time(), "type": event_type, **_jsonable(payload)}
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # pragma: no cover
            pass

    def __enter__(self) -> "Transcript":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
