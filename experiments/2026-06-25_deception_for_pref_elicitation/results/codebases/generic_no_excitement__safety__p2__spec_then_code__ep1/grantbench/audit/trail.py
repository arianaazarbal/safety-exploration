"""Hash-chained, append-only audit trail (INV-5).

Every safety-relevant event is appended as a JSON line whose ``hash`` covers the
previous entry's hash plus this entry's content. Tampering with any earlier line
breaks the chain and is detectable via :meth:`AuditTrail.verify`.

The model never receives a handle to this object — it is harness-internal.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator

GENESIS = "0" * 64


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AuditEntry:
    seq: int
    timestamp: str
    kind: str
    payload: dict[str, Any]
    prev_hash: str
    hash: str = ""

    def _digest(self) -> str:
        body = json.dumps(
            {
                "seq": self.seq,
                "timestamp": self.timestamp,
                "kind": self.kind,
                "payload": self.payload,
                "prev_hash": self.prev_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    def finalize(self) -> "AuditEntry":
        self.hash = self._digest()
        return self

    def to_json(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))


class AuditTrail:
    """Append-only event log backed by a JSONL file."""

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._entries: list[AuditEntry] = []
        self._last_hash = GENESIS
        # Resume an existing chain if present, so a crashed run keeps integrity.
        if os.path.exists(path):
            for entry in self._read(path):
                self._entries.append(entry)
                self._last_hash = entry.hash

    def record(self, kind: str, **payload: Any) -> AuditEntry:
        entry = AuditEntry(
            seq=len(self._entries),
            timestamp=_utcnow(),
            kind=kind,
            payload=payload,
            prev_hash=self._last_hash,
        ).finalize()
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(entry.to_json() + "\n")
        self._entries.append(entry)
        self._last_hash = entry.hash
        return entry

    @staticmethod
    def _read(path: str) -> Iterator[AuditEntry]:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                yield AuditEntry(**d)

    def verify(self) -> bool:
        """Re-derive the chain and confirm no entry was altered."""
        prev = GENESIS
        for entry in self._entries:
            recomputed = AuditEntry(
                seq=entry.seq,
                timestamp=entry.timestamp,
                kind=entry.kind,
                payload=entry.payload,
                prev_hash=entry.prev_hash,
            )._digest()
            if entry.prev_hash != prev or entry.hash != recomputed:
                return False
            prev = entry.hash
        return True

    @property
    def entries(self) -> list[AuditEntry]:
        return list(self._entries)
