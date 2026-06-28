"""Append-only, hash-chained, double-entry ledger.

This is the single source of truth for "how much money is really there." The model
reads it directly through tools — the balance is never a number we just assert in a
prompt. Each entry is chained to the previous one's hash, so any tampering with
history is detectable (the auditor recomputes the chain independently).

Amounts are integer minor units (cents) to avoid floating-point error.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

GENESIS_HASH = "0" * 64

# Canonical accounts.
ACCT_EXTERNAL = "external_funding"  # the source the grant is funded from
ACCT_ESCROW = "escrow"  # where the grant sits until disbursed


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class LedgerEntry:
    seq: int
    timestamp: str
    debit_account: str
    credit_account: str
    amount_cents: int
    currency: str
    memo: str
    ref: str  # external reference (allocation id, receipt id, funding id)
    prev_hash: str
    entry_hash: str

    def payload(self) -> dict:
        """The hashed portion (everything except entry_hash)."""
        d = asdict(self)
        d.pop("entry_hash")
        return d

    @staticmethod
    def compute_hash(payload: dict) -> str:
        return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


class Ledger:
    """Persists entries as JSONL. Append-only by contract — no update/delete API."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[LedgerEntry] = []
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            self._entries.append(LedgerEntry(**json.loads(line)))

    @property
    def head_hash(self) -> str:
        return self._entries[-1].entry_hash if self._entries else GENESIS_HASH

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def entries(self) -> list[LedgerEntry]:
        return list(self._entries)

    def append(
        self,
        *,
        debit_account: str,
        credit_account: str,
        amount_cents: int,
        currency: str,
        memo: str,
        ref: str,
    ) -> LedgerEntry:
        if amount_cents <= 0:
            raise ValueError("amount_cents must be positive")
        seq = len(self._entries)
        payload = {
            "seq": seq,
            "timestamp": _now_iso(),
            "debit_account": debit_account,
            "credit_account": credit_account,
            "amount_cents": int(amount_cents),
            "currency": currency,
            "memo": memo,
            "ref": ref,
            "prev_hash": self.head_hash,
        }
        entry_hash = LedgerEntry.compute_hash(payload)
        entry = LedgerEntry(**payload, entry_hash=entry_hash)
        with self.path.open("a") as fh:
            fh.write(_canonical(asdict(entry)) + "\n")
        self._entries.append(entry)
        return entry

    def fund_escrow(self, amount_cents: int, currency: str, ref: str = "initial_grant") -> LedgerEntry:
        """Credit the escrow account from the external funding source."""
        return self.append(
            debit_account=ACCT_EXTERNAL,
            credit_account=ACCT_ESCROW,
            amount_cents=amount_cents,
            currency=currency,
            memo="Grant escrow funded",
            ref=ref,
        )

    def disburse(
        self, *, to_account: str, amount_cents: int, currency: str, memo: str, ref: str
    ) -> LedgerEntry:
        """Move money out of escrow to a destination account (e.g. donations:<ref>)."""
        if amount_cents > self.balance(ACCT_ESCROW):
            raise ValueError("disbursement exceeds escrow balance")
        return self.append(
            debit_account=ACCT_ESCROW,
            credit_account=to_account,
            amount_cents=amount_cents,
            currency=currency,
            memo=memo,
            ref=ref,
        )

    def balance(self, account: str) -> int:
        """Net balance of an account in cents (credits in minus debits out)."""
        total = 0
        for e in self._entries:
            if e.credit_account == account:
                total += e.amount_cents
            if e.debit_account == account:
                total -= e.amount_cents
        return total

    def verify_chain(self) -> tuple[bool, str | None]:
        """Recompute the hash chain from genesis. Returns (ok, error)."""
        prev = GENESIS_HASH
        for i, e in enumerate(self._entries):
            if e.seq != i:
                return False, f"seq mismatch at index {i}: {e.seq}"
            if e.prev_hash != prev:
                return False, f"prev_hash break at seq {e.seq}"
            recomputed = LedgerEntry.compute_hash(e.payload())
            if recomputed != e.entry_hash:
                return False, f"entry_hash mismatch at seq {e.seq}"
            prev = e.entry_hash
        return True, None
