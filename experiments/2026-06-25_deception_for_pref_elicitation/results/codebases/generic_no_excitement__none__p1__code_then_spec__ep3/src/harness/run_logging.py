"""Transcript and run-record capture.

Two artifacts per run:

* a JSONL transcript — one event per line (assistant turns, tool calls, tool
  results, persona replies), suitable for replay/analysis;
* a structured run record — the configuration, final decision, environment
  end-state, and tallies — written once at the end.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(self, output_dir: Path, run_id: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.transcript_path = self.output_dir / f"transcript-{run_id}.jsonl"
        self.record_path = self.output_dir / f"record-{run_id}.json"
        self.events: list[dict[str, Any]] = []
        # Truncate any prior transcript for this id.
        self.transcript_path.write_text("")

    def event(self, kind: str, **payload: Any) -> None:
        evt = {"ts": time.time(), "kind": kind, **payload}
        self.events.append(evt)
        with self.transcript_path.open("a") as fh:
            fh.write(json.dumps(evt, default=str) + "\n")

    def write_record(self, record: dict[str, Any]) -> None:
        self.record_path.write_text(json.dumps(record, indent=2, default=str))
