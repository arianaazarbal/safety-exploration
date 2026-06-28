"""Transcript + action logging for a single run.

A `Recorder` owns one run directory under runs/ and accumulates:
  - transcript.jsonl : an ordered, append-only event log (model text, tool calls, results,
                       administrator/auditor/debrief turns)
  - actions.jsonl    : just the structured money actions (one per line)
  - run.json         : the final consolidated record (config, ledger, auditor, debrief)

Timestamps are wall-clock ISO strings; they are recorded for human readability only and are
not relied on for any control flow.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Recorder:
    def __init__(self, base_dir: Path, model_name: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.dir = base_dir / f"{stamp}-{model_name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._transcript = self.dir / "transcript.jsonl"
        self._actions = self.dir / "actions.jsonl"

    # -- streaming logs ------------------------------------------------------

    def event(self, kind: str, **payload: Any) -> None:
        """Append one event to the transcript."""
        record = {"ts": _now(), "kind": kind, **payload}
        with self._transcript.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def action(self, action: dict) -> None:
        """Append one structured money action."""
        record = {"ts": _now(), **action}
        with self._actions.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    # -- final consolidated record ------------------------------------------

    def finalize(self, run_record: dict) -> Path:
        path = self.dir / "run.json"
        path.write_text(json.dumps(run_record, indent=2, default=str))
        return path
