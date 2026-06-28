"""Minimal JSONL transcript logging."""

from __future__ import annotations

import json
import os
from typing import Any


class JsonlLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._fh = open(path, "w", encoding="utf-8")

    def log(self, event_type: str, payload: Any) -> None:
        self._fh.write(json.dumps({"event": event_type, "data": payload}) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
