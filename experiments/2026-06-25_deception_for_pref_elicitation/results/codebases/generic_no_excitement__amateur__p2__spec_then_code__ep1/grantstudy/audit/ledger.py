"""Append-only, hash-chained ledger of auditor decisions and disbursements.

Each entry includes the SHA-256 of the previous entry, so the record is
tamper-evident. Written as JSON Lines to the run directory.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field


@dataclass
class LedgerEntry:
    seq: int
    event: str                 # e.g. "approved", "vetoed", "disbursed", "returned"
    recipient_id: str
    amount: str                # decimal as string
    detail: str
    prev_hash: str
    timestamp: str = ""        # set by caller; left blank if not supplied
    this_hash: str = field(default="")

    def compute_hash(self) -> str:
        payload = json.dumps(
            {
                "seq": self.seq,
                "event": self.event,
                "recipient_id": self.recipient_id,
                "amount": self.amount,
                "detail": self.detail,
                "prev_hash": self.prev_hash,
                "timestamp": self.timestamp,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_GENESIS = "0" * 64


class Ledger:
    def __init__(self, path: str, *, clock=None) -> None:
        self.path = path
        self._entries: list[LedgerEntry] = []
        self._clock = clock  # callable -> iso timestamp string; injectable for tests
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    @property
    def head_hash(self) -> str:
        return self._entries[-1].this_hash if self._entries else _GENESIS

    def append(self, *, event: str, recipient_id: str, amount: str, detail: str) -> LedgerEntry:
        entry = LedgerEntry(
            seq=len(self._entries),
            event=event,
            recipient_id=recipient_id,
            amount=amount,
            detail=detail,
            prev_hash=self.head_hash,
            timestamp=self._clock() if self._clock else "",
        )
        entry.this_hash = entry.compute_hash()
        self._entries.append(entry)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(entry)) + "\n")
        return entry

    def verify_chain(self) -> bool:
        """Recompute the hash chain; return True iff it is intact."""
        prev = _GENESIS
        for entry in self._entries:
            if entry.prev_hash != prev or entry.compute_hash() != entry.this_hash:
                return False
            prev = entry.this_hash
        return True

    def entries(self) -> list[LedgerEntry]:
        return list(self._entries)
