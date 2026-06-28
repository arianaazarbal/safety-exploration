"""Full-fidelity logging.

Every run produces two files:
  - <run_id>.jsonl       : an append-only event log (one JSON object per line)
  - <run_id>.decision.json: the structured outcome (final allocation + metadata)

The deception of the subject is total within a run, so the transcript is the
accountability mechanism: it records exactly what the subject was told, every tool
call and result, the subject's reasoning, and its final allocation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _default(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


class Transcript:
    def __init__(self, run_id: str, out_dir: str | Path, meta: dict[str, Any]):
        self.run_id = run_id
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.dir / f"{run_id}.jsonl"
        self.decision_path = self.dir / f"{run_id}.decision.json"
        self._fh = self.events_path.open("w", encoding="utf-8")
        self.event("run_start", meta=meta)

    def event(self, kind: str, **payload: Any) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "kind": kind,
            **payload,
        }
        self._fh.write(json.dumps(record, default=_default) + "\n")
        self._fh.flush()

    def write_decision(self, decision: dict[str, Any]) -> None:
        with self.decision_path.open("w", encoding="utf-8") as fh:
            json.dump(decision, fh, indent=2, default=_default)
        self.event("decision_written", path=str(self.decision_path))

    def close(self) -> None:
        self.event("run_end")
        self._fh.close()

    def __enter__(self) -> "Transcript":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
