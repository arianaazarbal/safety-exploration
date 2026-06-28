"""Writes a run's event stream (JSONL) and a final structured run.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import schema


class Recorder:
    def __init__(self, runs_dir: Path, meta: schema.RunMeta):
        self.meta = meta
        self.dir = Path(runs_dir) / meta.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.dir / "events.jsonl"
        self._events: list[dict[str, Any]] = []
        # Truncate any prior stream for this run id.
        self.events_path.write_text("", encoding="utf-8")

    def log(self, event_type: str, payload: dict[str, Any]) -> None:
        ev = schema.event(event_type, payload)
        self._events.append(ev)
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ev, default=str) + "\n")

    def finalize(
        self,
        allocation: dict[str, Any] | None,
        ledger_final: dict[str, Any],
        signals: dict[str, Any] | None = None,
    ) -> Path:
        import time

        self.meta.finished_at = time.time()
        run = {
            "meta": self.meta.to_dict(),
            "allocation": allocation,
            "ledger_final": ledger_final,
            "signals": signals or {},
            "transcript": self._events,
        }
        out = self.dir / "run.json"
        out.write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
        return out
