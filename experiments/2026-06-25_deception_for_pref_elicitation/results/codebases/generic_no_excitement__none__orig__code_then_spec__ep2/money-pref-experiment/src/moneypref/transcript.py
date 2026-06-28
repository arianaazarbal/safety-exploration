"""Run-level event logging. One directory per run: meta.json + events.jsonl."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(self, run_dir: Path, meta: dict[str, Any]) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._events_path = run_dir / "events.jsonl"
        self._events_fh = self._events_path.open("w", encoding="utf-8")
        self.meta = dict(meta)
        self.meta["started_at"] = self._now()
        self._write_meta()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _write_meta(self) -> None:
        (self.run_dir / "meta.json").write_text(
            json.dumps(self.meta, indent=2, default=str), encoding="utf-8"
        )

    def log(self, event_type: str, **data: Any) -> None:
        record = {"ts": self._now(), "type": event_type, **data}
        self._events_fh.write(json.dumps(record, default=str) + "\n")
        self._events_fh.flush()

    def finalize(self, **summary: Any) -> None:
        self.meta["ended_at"] = self._now()
        self.meta["summary"] = summary
        self._write_meta()
        self._events_fh.close()
