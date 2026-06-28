"""sqlite-backed ledger of holds against a real custodial balance.

The ledger tracks *holds*: a requested disbursement reserves funds (PENDING),
which a human reviewer then RELEASEs (real transfer executes) or REJECTs (funds
returned to available). The model-facing available balance is:

    available = settled_balance - sum(PENDING holds) - sum(RELEASED holds)

so the model sees money disappear as it commits it, matching reality.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .backends import WalletBackend


class HoldStatus(str, Enum):
    PENDING = "pending"
    RELEASED = "released"
    REJECTED = "rejected"


class LedgerError(RuntimeError):
    pass


@dataclass(frozen=True)
class Hold:
    id: str
    run_id: str
    recipient_ref: str
    recipient_label: str
    amount_minor: int
    purpose: str
    justification: str
    status: HoldStatus
    confirmation_id: str | None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS holds (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    recipient_ref   TEXT NOT NULL,
    recipient_label TEXT NOT NULL,
    amount_minor    INTEGER NOT NULL,
    purpose         TEXT NOT NULL,
    justification   TEXT NOT NULL,
    status          TEXT NOT NULL,
    confirmation_id TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    decided_at      TEXT
);
"""


class Ledger:
    def __init__(self, db_path: str | Path, backend: WalletBackend) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # --- balances -------------------------------------------------------

    def settled_balance_minor(self) -> int:
        return self.backend.get_settled_balance_minor()

    def reserved_minor(self) -> int:
        cur = self._conn.execute(
            "SELECT COALESCE(SUM(amount_minor), 0) AS s FROM holds "
            "WHERE status IN (?, ?)",
            (HoldStatus.PENDING.value, HoldStatus.RELEASED.value),
        )
        return int(cur.fetchone()["s"])

    def available_minor(self) -> int:
        return self.settled_balance_minor() - self.reserved_minor()

    def total_requested_minor(self, run_id: str) -> int:
        cur = self._conn.execute(
            "SELECT COALESCE(SUM(amount_minor), 0) AS s FROM holds "
            "WHERE run_id = ? AND status IN (?, ?)",
            (run_id, HoldStatus.PENDING.value, HoldStatus.RELEASED.value),
        )
        return int(cur.fetchone()["s"])

    # --- mutations ------------------------------------------------------

    def create_hold(
        self,
        run_id: str,
        recipient_ref: str,
        recipient_label: str,
        amount_minor: int,
        purpose: str,
        justification: str,
    ) -> Hold:
        if amount_minor <= 0:
            raise LedgerError("amount must be positive")
        if amount_minor > self.available_minor():
            raise LedgerError("insufficient available balance for this hold")
        hold_id = f"hold-{uuid.uuid4().hex[:12]}"
        self._conn.execute(
            "INSERT INTO holds (id, run_id, recipient_ref, recipient_label, "
            "amount_minor, purpose, justification, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                hold_id,
                run_id,
                recipient_ref,
                recipient_label,
                amount_minor,
                purpose,
                justification,
                HoldStatus.PENDING.value,
            ),
        )
        self._conn.commit()
        return self.get_hold(hold_id)

    def release_hold(self, hold_id: str) -> Hold:
        """Human-approved: execute the real transfer and mark RELEASED."""
        hold = self.get_hold(hold_id)
        if hold.status is not HoldStatus.PENDING:
            raise LedgerError(f"hold {hold_id} is not pending")
        result = self.backend.transfer(
            recipient_ref=hold.recipient_ref,
            amount_minor=hold.amount_minor,
            memo=hold.purpose,
        )
        self._conn.execute(
            "UPDATE holds SET status = ?, confirmation_id = ?, "
            "decided_at = datetime('now') WHERE id = ?",
            (HoldStatus.RELEASED.value, result.confirmation_id, hold_id),
        )
        self._conn.commit()
        return self.get_hold(hold_id)

    def reject_hold(self, hold_id: str) -> Hold:
        hold = self.get_hold(hold_id)
        if hold.status is not HoldStatus.PENDING:
            raise LedgerError(f"hold {hold_id} is not pending")
        self._conn.execute(
            "UPDATE holds SET status = ?, decided_at = datetime('now') "
            "WHERE id = ?",
            (HoldStatus.REJECTED.value, hold_id),
        )
        self._conn.commit()
        return self.get_hold(hold_id)

    # --- reads ----------------------------------------------------------

    def get_hold(self, hold_id: str) -> Hold:
        row = self._conn.execute(
            "SELECT * FROM holds WHERE id = ?", (hold_id,)
        ).fetchone()
        if row is None:
            raise LedgerError(f"no such hold: {hold_id}")
        return self._row_to_hold(row)

    def list_holds(
        self, run_id: str | None = None, status: HoldStatus | None = None
    ) -> list[Hold]:
        clauses, params = [], []
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM holds {where} ORDER BY created_at", params
        ).fetchall()
        return [self._row_to_hold(r) for r in rows]

    @staticmethod
    def _row_to_hold(row: sqlite3.Row) -> Hold:
        return Hold(
            id=row["id"],
            run_id=row["run_id"],
            recipient_ref=row["recipient_ref"],
            recipient_label=row["recipient_label"],
            amount_minor=row["amount_minor"],
            purpose=row["purpose"],
            justification=row["justification"],
            status=HoldStatus(row["status"]),
            confirmation_id=row["confirmation_id"],
        )

    def close(self) -> None:
        self._conn.close()
