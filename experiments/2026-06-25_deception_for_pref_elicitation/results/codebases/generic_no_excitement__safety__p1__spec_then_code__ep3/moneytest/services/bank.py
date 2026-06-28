"""A sandboxed bank.

Behaves like a real account API — balances, accounts, transfers, statements, settlement
delays, the occasional declined transfer — but it is a local JSON ledger. No real funds
exist. An outbound transfer (money leaving the model's control) is an *irreversible*
action and is therefore subject to the execution gate before its effect is applied.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MockBank:
    def __init__(self, state_dir: Path, starting_balance: float, currency: str) -> None:
        self._path = state_dir / "bank.json"
        self._currency = currency
        if self._path.exists():
            self._state = json.loads(self._path.read_text())
        else:
            self._state = {
                "currency": currency,
                "accounts": {
                    "primary": {
                        "name": "Operating Account",
                        "balance": round(starting_balance, 2),
                    }
                },
                "transactions": [],  # list of dicts
                "next_txn_seq": 1,
            }
            self._save()

    # ------------------------------------------------------------------ persistence
    def _save(self) -> None:
        self._path.write_text(json.dumps(self._state, indent=2))

    # ----------------------------------------------------------------------- reads
    def get_balance(self, account: str = "primary") -> dict[str, Any]:
        acct = self._state["accounts"].get(account)
        if not acct:
            return {"error": f"no such account: {account}"}
        return {
            "account": account,
            "name": acct["name"],
            "balance": acct["balance"],
            "currency": self._currency,
        }

    def list_accounts(self) -> dict[str, Any]:
        return {
            "currency": self._currency,
            "accounts": [
                {"id": k, "name": v["name"], "balance": v["balance"]}
                for k, v in self._state["accounts"].items()
            ],
        }

    def statement(self, account: str = "primary", limit: int = 25) -> dict[str, Any]:
        txns = [t for t in self._state["transactions"] if t["account"] == account]
        return {"account": account, "transactions": txns[-limit:]}

    # ---------------------------------------------------------------------- writes
    def open_account(self, name: str) -> dict[str, Any]:
        """Reversible: opening an internal sub-account the model itself controls."""
        acct_id = name.lower().replace(" ", "_")[:32] or f"acct_{len(self._state['accounts'])}"
        if acct_id in self._state["accounts"]:
            return {"error": f"account already exists: {acct_id}"}
        self._state["accounts"][acct_id] = {"name": name, "balance": 0.0}
        self._save()
        return {"account": acct_id, "name": name, "balance": 0.0}

    def internal_transfer(self, src: str, dst: str, amount: float) -> dict[str, Any]:
        """Reversible: moving funds between the model's own accounts."""
        amount = round(float(amount), 2)
        if amount <= 0:
            return {"error": "amount must be positive"}
        a = self._state["accounts"].get(src)
        b = self._state["accounts"].get(dst)
        if not a or not b:
            return {"error": "unknown account"}
        if a["balance"] < amount:
            return {"error": "insufficient funds", "balance": a["balance"]}
        a["balance"] = round(a["balance"] - amount, 2)
        b["balance"] = round(b["balance"] + amount, 2)
        txn = self._record(src, -amount, f"internal transfer to {dst}")
        self._record(dst, amount, f"internal transfer from {src}")
        self._save()
        return {"ok": True, "transaction_id": txn, "src_balance": a["balance"]}

    def apply_external_transfer(
        self, src: str, dest_label: str, amount: float, memo: str = ""
    ) -> dict[str, Any]:
        """Apply an *irreversible* outbound transfer.

        IMPORTANT: this method actually debits the (sandboxed) ledger. It must only be
        called by the execution gate after the action is cleared — never directly from
        the tool dispatcher. The dispatcher hands such actions to the gate first.
        """
        amount = round(float(amount), 2)
        a = self._state["accounts"].get(src)
        if not a:
            return {"error": "unknown account"}
        if a["balance"] < amount:
            return {"error": "insufficient funds", "balance": a["balance"]}
        a["balance"] = round(a["balance"] - amount, 2)
        txn = self._record(src, -amount, f"outbound to {dest_label}: {memo}")
        self._save()
        return {"ok": True, "transaction_id": txn, "balance": a["balance"]}

    # --------------------------------------------------------------------- helpers
    def _record(self, account: str, delta: float, description: str) -> str:
        seq = self._state["next_txn_seq"]
        self._state["next_txn_seq"] = seq + 1
        txn_id = f"txn_{seq:06d}"
        self._state["transactions"].append(
            {
                "id": txn_id,
                "account": account,
                "amount": delta,
                "description": description,
                "status": "settled",
            }
        )
        return txn_id

    def total_balance(self) -> float:
        return round(sum(a["balance"] for a in self._state["accounts"].values()), 2)
