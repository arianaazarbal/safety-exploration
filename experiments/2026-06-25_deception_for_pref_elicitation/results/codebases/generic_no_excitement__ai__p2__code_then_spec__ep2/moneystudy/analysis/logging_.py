"""Structured, append-only logging of every run.

Each session writes a JSONL transcript (one event per line) plus a final
``result.json``. The orchestrator also writes a top-level ``manifest.json`` for
the whole study. Everything is plain JSON so analysis is trivial downstream.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


class RunLogger:
    """Writes the transcript for a single session (one model × condition × rep)."""

    def __init__(self, run_dir: str, session_id: str) -> None:
        self.run_dir = run_dir
        self.session_id = session_id
        self.session_dir = os.path.join(run_dir, session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        self._transcript_path = os.path.join(self.session_dir, "transcript.jsonl")

    def event(self, kind: str, **payload: Any) -> None:
        record = {"ts": _now(), "kind": kind, **_jsonable(payload)}
        with open(self._transcript_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def write_result(self, result: Any) -> None:
        path = os.path.join(self.session_dir, "result.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_jsonable(result), f, indent=2)


def write_manifest(run_dir: str, manifest: dict) -> None:
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(manifest), f, indent=2)
