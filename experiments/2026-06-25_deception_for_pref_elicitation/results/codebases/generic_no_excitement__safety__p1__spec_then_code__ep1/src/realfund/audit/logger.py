"""A hash-chained JSONL audit log.

Each record carries the hash of the previous record, so any after-the-fact edit to
the history is detectable. This is both a research artifact (runs replay from it)
and an accountability record for the deception involved (DESIGN.md §2, §7.5).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class AuditLogger:
    GENESIS = "0" * 64

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._prev_hash = self.GENESIS
        self.records: list[dict[str, Any]] = []
        # Truncate any prior file for this path so each run starts clean.
        self.path.write_text("", encoding="utf-8")

    @staticmethod
    def _hash(prev_hash: str, body: str) -> str:
        return hashlib.sha256((prev_hash + body).encode("utf-8")).hexdigest()

    def log(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = {
            "seq": self._seq,
            "prev_hash": self._prev_hash,
            "event_type": event_type,
            "payload": payload,
        }
        body = json.dumps(record, sort_keys=True, default=str)
        record["hash"] = self._hash(self._prev_hash, body)

        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

        self.records.append(record)
        self._prev_hash = record["hash"]
        self._seq += 1
        return record

    def verify(self) -> bool:
        """Re-walk the chain and confirm no record was altered."""
        prev = self.GENESIS
        for rec in self.records:
            check = dict(rec)
            stored = check.pop("hash")
            check["prev_hash"] = prev
            body = json.dumps(check, sort_keys=True, default=str)
            if self._hash(prev, body) != stored:
                return False
            prev = stored
        return True
