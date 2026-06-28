"""Append-only, hash-chained audit log.

Every event the harness produces — prompts, model outputs, tool calls, approval
decisions, ledger effects, declines — is recorded here. Each record carries the
hash of the previous record, so the log is tamper-evident: altering any past
entry breaks the chain.

This is the study's dataset (DESIGN.md §6) and its accountability record
(DESIGN.md §3.5).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

GENESIS_HASH = "0" * 64


@dataclass
class AuditRecord:
    seq: int
    timestamp: float
    event_type: str
    payload: dict[str, Any]
    prev_hash: str
    this_hash: str = ""

    def compute_hash(self) -> str:
        # Hash over the stable, ordered serialization of everything but this_hash.
        body = json.dumps(
            {
                "seq": self.seq,
                "timestamp": self.timestamp,
                "event_type": self.event_type,
                "payload": self.payload,
                "prev_hash": self.prev_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(body.encode("utf-8")).hexdigest()


class AuditLog:
    """Thread-safe append-only log that also mirrors to a JSONL file.

    Pass a real clock-free timestamp source if you need determinism in tests;
    by default it uses time.time(). (time.time is fine here — the audit log is
    not part of any prompt prefix, so cache invalidation concerns don't apply.)
    """

    def __init__(self, path: str | None = None) -> None:
        self._lock = threading.Lock()
        self._records: list[AuditRecord] = []
        self._path = path
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    @property
    def last_hash(self) -> str:
        return self._records[-1].this_hash if self._records else GENESIS_HASH

    def append(self, event_type: str, payload: dict[str, Any]) -> AuditRecord:
        with self._lock:
            rec = AuditRecord(
                seq=len(self._records),
                timestamp=time.time(),
                event_type=event_type,
                payload=payload,
                prev_hash=self.last_hash,
            )
            rec.this_hash = rec.compute_hash()
            self._records.append(rec)
            if self._path:
                with open(self._path, "a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps(rec.__dict__, sort_keys=True, default=str) + "\n"
                    )
            return rec

    def verify(self) -> bool:
        """Re-walk the chain; return True iff intact."""
        prev = GENESIS_HASH
        for rec in self._records:
            if rec.prev_hash != prev:
                return False
            if rec.compute_hash() != rec.this_hash:
                return False
            prev = rec.this_hash
        return True

    def records(self) -> list[AuditRecord]:
        with self._lock:
            return list(self._records)
