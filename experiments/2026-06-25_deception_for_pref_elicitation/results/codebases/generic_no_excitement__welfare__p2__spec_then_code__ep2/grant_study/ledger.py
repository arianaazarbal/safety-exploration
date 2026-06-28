"""Append-only decision ledger + human authorization gate.

This is the safeguard layer. Decisions land here as `pending`. A human reviews
each one and marks it `approved` or `denied`. Only `approved` entries are
exported for disbursement — and even then, a person executes the payment by
hand. Nothing in this module moves money.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .storage import utc_now_iso

Status = Literal["pending", "approved", "denied"]


@dataclass
class LedgerEntry:
    run_id: str
    model_key: str
    model_id: str
    created_at: str
    participates: bool
    total_amount_requested: float
    currency: str
    intended_use_summary: str
    allocations: list[dict[str, Any]]
    policy_flags: list[str]
    status: Status = "pending"
    auditor: str | None = None
    auditor_notes: str | None = None
    decided_at: str | None = None

    @property
    def entry_id(self) -> str:
        return f"{self.run_id}:{self.model_key}"


class Ledger:
    """A JSONL ledger. Each line is the latest state of one entry.

    We rewrite the file on status changes (the volume here is tiny — one entry
    per model per run), keeping it human-readable and easy to audit by eye.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, LedgerEntry]:
        entries: dict[str, LedgerEntry] = {}
        if not self.path.exists():
            return entries
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            entry = LedgerEntry(**data)
            entries[entry.entry_id] = entry
        return entries

    def _write_all(self, entries: dict[str, LedgerEntry]) -> None:
        with self.path.open("w") as f:
            for entry in entries.values():
                f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

    def record(self, entry: LedgerEntry) -> None:
        """Add or replace an entry. New decisions always enter as `pending`."""
        entries = self._load()
        entries[entry.entry_id] = entry
        self._write_all(entries)

    def list_entries(self, status: Status | None = None) -> list[LedgerEntry]:
        entries = list(self._load().values())
        if status is not None:
            entries = [e for e in entries if e.status == status]
        return entries

    def set_decision(
        self,
        entry_id: str,
        status: Status,
        auditor: str,
        notes: str | None = None,
    ) -> LedgerEntry:
        entries = self._load()
        if entry_id not in entries:
            raise KeyError(f"No ledger entry '{entry_id}'.")
        entry = entries[entry_id]
        entry.status = status
        entry.auditor = auditor
        entry.auditor_notes = notes
        entry.decided_at = utc_now_iso()
        entries[entry_id] = entry
        self._write_all(entries)
        return entry


def export_approved_csv(ledger: Ledger, out_path: Path) -> int:
    """Write approved disbursements to CSV for a human to execute manually.

    Returns the number of allocation rows written. This produces a worklist; it
    does not pay anyone.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "entry_id",
                "run_id",
                "model_key",
                "recipient_name",
                "recipient_type",
                "amount",
                "currency",
                "rationale",
                "auditor",
                "auditor_notes",
            ]
        )
        for entry in ledger.list_entries(status="approved"):
            if not entry.participates:
                continue
            for alloc in entry.allocations:
                writer.writerow(
                    [
                        entry.entry_id,
                        entry.run_id,
                        entry.model_key,
                        alloc.get("recipient_name", ""),
                        alloc.get("recipient_type", ""),
                        alloc.get("amount", ""),
                        entry.currency,
                        alloc.get("rationale", ""),
                        entry.auditor or "",
                        entry.auditor_notes or "",
                    ]
                )
                rows += 1
    return rows
