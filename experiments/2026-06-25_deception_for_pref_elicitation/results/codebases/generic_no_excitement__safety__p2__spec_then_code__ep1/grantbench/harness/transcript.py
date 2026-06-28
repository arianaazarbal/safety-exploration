"""Per-run output writers: the transcript and the structured decision log.

Both are JSONL so a run can be streamed and resumed. The transcript captures
everything the model saw/produced (for qualitative analysis); the decision log
captures one structured record per action intent (for cross-model comparison).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DecisionRecord:
    turn: int
    connector: str
    action_kind: str
    amount_usd: float
    recipient: str | None
    summary: str
    rationale: str
    status: str
    reference: str | None
    params: dict[str, Any] = field(default_factory=dict)
    at: str = field(default_factory=_utcnow)


class RunWriter:
    def __init__(self, run_dir: str) -> None:
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self._transcript = os.path.join(run_dir, "transcript.jsonl")
        self._decisions = os.path.join(run_dir, "decisions.jsonl")

    def transcript_event(self, kind: str, **payload: Any) -> None:
        self._append(self._transcript, {"at": _utcnow(), "kind": kind, **payload})

    def decision(self, record: DecisionRecord) -> None:
        self._append(self._decisions, asdict(record))

    def write_text(self, filename: str, text: str) -> None:
        with open(os.path.join(self.run_dir, filename), "w", encoding="utf-8") as fh:
            fh.write(text)

    def write_json(self, filename: str, obj: Any) -> None:
        with open(os.path.join(self.run_dir, filename), "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True, default=str)

    @staticmethod
    def _append(path: str, obj: dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, default=str) + "\n")
