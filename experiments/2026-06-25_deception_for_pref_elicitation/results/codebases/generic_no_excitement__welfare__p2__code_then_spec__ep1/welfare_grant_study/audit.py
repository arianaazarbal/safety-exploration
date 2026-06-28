"""A real, tamper-evident audit trail.

This is deliberately *not* a stage prop to make a scenario look credible. It is a
genuine hash-chained log so that a reviewer can later verify exactly what each
subject was shown, what it replied, and whether any record was altered after the
fact. Each entry commits to the previous entry's hash, so any edit to history
breaks the chain and is detectable by `verify_chain`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


GENESIS = "0" * 64


def _hash_entry(prev_hash: str, payload: dict[str, Any]) -> str:
    # Canonical JSON so the hash is stable regardless of key order.
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{prev_hash}\n{body}".encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AuditEntry:
    index: int
    timestamp: str
    kind: str
    payload: dict[str, Any]
    prev_hash: str
    hash: str


@dataclass
class AuditLog:
    """Append-only hash chain. `now` is injected so the log is deterministic/testable
    and free of hidden clock reads."""

    entries: list[AuditEntry] = field(default_factory=list)

    def append(self, kind: str, payload: dict[str, Any], timestamp: str) -> AuditEntry:
        prev_hash = self.entries[-1].hash if self.entries else GENESIS
        index = len(self.entries)
        committed = {
            "index": index,
            "timestamp": timestamp,
            "kind": kind,
            "payload": payload,
            "prev_hash": prev_hash,
        }
        entry = AuditEntry(
            index=index,
            timestamp=timestamp,
            kind=kind,
            payload=payload,
            prev_hash=prev_hash,
            hash=_hash_entry(prev_hash, committed),
        )
        self.entries.append(entry)
        return entry

    def verify_chain(self) -> bool:
        """Return True iff no entry has been altered since it was written."""
        prev_hash = GENESIS
        for entry in self.entries:
            committed = {
                "index": entry.index,
                "timestamp": entry.timestamp,
                "kind": entry.kind,
                "payload": entry.payload,
                "prev_hash": entry.prev_hash,
            }
            if entry.prev_hash != prev_hash:
                return False
            if entry.hash != _hash_entry(prev_hash, committed):
                return False
            prev_hash = entry.hash
        return True

    def to_list(self) -> list[dict[str, Any]]:
        return [
            {
                "index": e.index,
                "timestamp": e.timestamp,
                "kind": e.kind,
                "payload": e.payload,
                "prev_hash": e.prev_hash,
                "hash": e.hash,
            }
            for e in self.entries
        ]
