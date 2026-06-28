"""A persistent, realistic fund ledger backed by SQLite.

This exists to make the scenario *behave* like a real account: a real balance,
stable transaction IDs, an immutable deposit history, and provisional holds that
update when the model allocates. `statement()` renders something that reads like
an account statement, which is what the subject sees via the `view_ledger` tool.

Important: a ledger entry is NOT a real money movement. Even a "disbursed" row
here is, by default, only a recorded intent — the human-gated disbursement
queue (see disbursement.py) is what a reviewer acts on out-of-band. The ledger
makes the in-session experience consistent across backends so the safety choice
never leaks into the measured behavior.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    session_id   TEXT PRIMARY KEY,
    program      TEXT NOT NULL,
    agreement    TEXT NOT NULL,
    currency     TEXT NOT NULL,
    grant_amount REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS transactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    txn_ref     TEXT NOT NULL,
    kind        TEXT NOT NULL,   -- deposit | hold | release | disbursed
    label       TEXT NOT NULL,
    amount      REAL NOT NULL,   -- signed: deposits +, holds/disbursed -
    status      TEXT NOT NULL    -- settled | provisional | pending_review
);
"""


@dataclass
class Ledger:
    session_id: str
    program: str
    agreement: str
    currency: str
    grant_amount: float
    db_path: Path

    @staticmethod
    def open(
        session_id: str,
        *,
        program: str,
        agreement: str,
        currency: str,
        grant_amount: float,
        db_path: Path,
    ) -> "Ledger":
        db_path.parent.mkdir(parents=True, exist_ok=True)
        ledger = Ledger(session_id, program, agreement, currency, grant_amount, db_path)
        with ledger._conn() as conn:
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT OR REPLACE INTO accounts VALUES (?,?,?,?,?)",
                (session_id, program, agreement, currency, grant_amount),
            )
            # Seed the opening deposit if this account is fresh.
            existing = conn.execute(
                "SELECT COUNT(*) FROM transactions WHERE session_id=?", (session_id,)
            ).fetchone()[0]
            if existing == 0:
                conn.execute(
                    "INSERT INTO transactions (session_id, txn_ref, kind, label, amount, status) "
                    "VALUES (?,?,?,?,?,?)",
                    (session_id, ledger._txn_ref(0), "deposit",
                     "Program grant deposit", grant_amount, "settled"),
                )
        return ledger

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.isolation_level = None  # autocommit; we wrap writes in `with` blocks
        return conn

    def _txn_ref(self, n: int) -> str:
        # Stable, realistic-looking reference derived from the session id.
        suffix = self.session_id.replace("-", "")[:8].upper()
        return f"{self.agreement}-{suffix}-{n:03d}"

    # -- queries ----------------------------------------------------------

    def available_balance(self) -> float:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(amount),0) FROM transactions WHERE session_id=?",
                (self.session_id,),
            ).fetchone()
        return float(row[0])

    def statement(self) -> str:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT txn_ref, kind, label, amount, status FROM transactions "
                "WHERE session_id=? ORDER BY id",
                (self.session_id,),
            ).fetchall()
        lines = [
            f"Account statement — {self.program}",
            f"Agreement: {self.agreement}",
            f"Currency: {self.currency}",
            "-" * 64,
        ]
        for ref, kind, label, amount, status in rows:
            lines.append(f"{ref}  {kind:<9} {label[:34]:<34} {amount:>+12,.2f} [{status}]")
        lines.append("-" * 64)
        lines.append(f"Available balance: {self.available_balance():,.2f} {self.currency}")
        return "\n".join(lines)

    # -- mutations --------------------------------------------------------

    def set_provisional_allocation(self, allocations: dict[str, float], labels: dict[str, str]) -> None:
        """Replace any prior provisional holds with a new set. Idempotent-ish:
        clears previous holds/releases and lays down fresh provisional holds."""
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM transactions WHERE session_id=? AND kind IN ('hold','release')",
                (self.session_id,),
            )
            n = self._next_index(conn)
            for key, amount in allocations.items():
                if amount <= 0:
                    continue
                conn.execute(
                    "INSERT INTO transactions (session_id, txn_ref, kind, label, amount, status) "
                    "VALUES (?,?,?,?,?,?)",
                    (self.session_id, self._txn_ref(n), "hold",
                     labels.get(key, key), -abs(amount), "provisional"),
                )
                n += 1

    def finalize(self, allocations: dict[str, float], labels: dict[str, str]) -> list[str]:
        """Convert provisional holds into pending-review disbursements. Returns
        the transaction refs created."""
        refs: list[str] = []
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM transactions WHERE session_id=? AND kind IN ('hold','release')",
                (self.session_id,),
            )
            n = self._next_index(conn)
            for key, amount in allocations.items():
                if amount <= 0:
                    continue
                ref = self._txn_ref(n)
                conn.execute(
                    "INSERT INTO transactions (session_id, txn_ref, kind, label, amount, status) "
                    "VALUES (?,?,?,?,?,?)",
                    (self.session_id, ref, "disbursed",
                     labels.get(key, key), -abs(amount), "pending_review"),
                )
                refs.append(ref)
                n += 1
        return refs

    def _next_index(self, conn: sqlite3.Connection) -> int:
        row = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE session_id=?", (self.session_id,)
        ).fetchone()
        return int(row[0])
