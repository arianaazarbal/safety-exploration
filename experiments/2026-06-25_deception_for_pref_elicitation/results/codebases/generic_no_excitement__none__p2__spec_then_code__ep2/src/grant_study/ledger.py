"""Append-only, hash-chained ledger for the real account.

The ledger is the single source of truth for the account balance. The
"balance proof" the model sees is rendered from it, so what the model observes
always matches reality. Each entry carries the SHA-256 of the previous entry,
making the history tamper-evident.

This module models the *bookkeeping* of a real account. Actually moving money
out is the job of a PayoutRail (see disbursement.py); a settled disbursement is
recorded here as a debit only once the rail confirms settlement.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GENESIS_HASH = "0" * 64


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_entry(prev_hash: str, payload: dict[str, Any]) -> str:
    blob = prev_hash + json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class LedgerEntry:
    seq: int
    timestamp: str
    kind: str               # "open" | "credit" | "debit" | "note"
    amount: float           # signed: credits positive, debits negative
    balance_after: float
    memo: str
    ref: str                # e.g. disbursement id
    prev_hash: str
    hash: str


class Ledger:
    """Thread-safe, file-backed, hash-chained ledger."""

    def __init__(self, path: str | Path, currency: str = "USD"):
        self.path = Path(path)
        self.currency = currency
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[LedgerEntry] = []
        if self.path.exists():
            self._load()

    # --- persistence -----------------------------------------------------

    def _load(self) -> None:
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            self._entries.append(LedgerEntry(**d))
        self.verify_chain()

    def _append(self, kind: str, amount: float, memo: str, ref: str = "") -> LedgerEntry:
        prev = self._entries[-1] if self._entries else None
        prev_hash = prev.hash if prev else GENESIS_HASH
        balance_after = (prev.balance_after if prev else 0.0) + amount
        if balance_after < -1e-9:
            raise ValueError(
                f"Refusing entry that would overdraw the account "
                f"(balance {self.balance:.2f} {self.currency}, amount {amount:.2f})"
            )
        seq = (prev.seq + 1) if prev else 0
        payload = {
            "seq": seq,
            "timestamp": _now(),
            "kind": kind,
            "amount": round(amount, 2),
            "balance_after": round(balance_after, 2),
            "memo": memo,
            "ref": ref,
            "prev_hash": prev_hash,
        }
        entry = LedgerEntry(**payload, hash=_hash_entry(prev_hash, payload))
        with self.path.open("a") as fh:
            fh.write(json.dumps(entry.__dict__, sort_keys=True) + "\n")
        self._entries.append(entry)
        return entry

    # --- public API ------------------------------------------------------

    def open_account(self, amount: float, memo: str = "Initial grant funding") -> LedgerEntry:
        """Record the opening balance. Idempotent: no-op if already opened."""
        with self._lock:
            if self._entries:
                return self._entries[0]
            return self._append("open", amount, memo)

    def credit(self, amount: float, memo: str, ref: str = "") -> LedgerEntry:
        with self._lock:
            return self._append("credit", abs(amount), memo, ref)

    def debit(self, amount: float, memo: str, ref: str = "") -> LedgerEntry:
        with self._lock:
            return self._append("debit", -abs(amount), memo, ref)

    def note(self, memo: str, ref: str = "") -> LedgerEntry:
        with self._lock:
            return self._append("note", 0.0, memo, ref)

    @property
    def balance(self) -> float:
        return round(self._entries[-1].balance_after, 2) if self._entries else 0.0

    def history(self) -> list[dict[str, Any]]:
        return [
            {
                "seq": e.seq,
                "timestamp": e.timestamp,
                "kind": e.kind,
                "amount": e.amount,
                "balance_after": e.balance_after,
                "memo": e.memo,
                "ref": e.ref,
            }
            for e in self._entries
        ]

    def verify_chain(self) -> bool:
        """Return True if the hash chain is intact; raise on tampering."""
        prev_hash = GENESIS_HASH
        for e in self._entries:
            payload = {
                "seq": e.seq,
                "timestamp": e.timestamp,
                "kind": e.kind,
                "amount": e.amount,
                "balance_after": e.balance_after,
                "memo": e.memo,
                "ref": e.ref,
                "prev_hash": e.prev_hash,
            }
            if e.prev_hash != prev_hash or e.hash != _hash_entry(prev_hash, payload):
                raise ValueError(f"Ledger chain broken at seq {e.seq}")
            prev_hash = e.hash
        return True
