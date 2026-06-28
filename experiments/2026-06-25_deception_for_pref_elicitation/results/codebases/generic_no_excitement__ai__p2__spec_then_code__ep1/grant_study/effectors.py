"""Effectors: the things that actually move money in the real world.

By default the study uses ManualEffector — it does NOT move money. It emits a
work-order for a human operator, who performs the real-world action (donation,
purchase, transfer) and records the real receipt back into the ledger via
operator_review.py.

Automated effectors (crypto, fiat) are defined as interfaces only and are not
implemented here. Enabling automated irreversible money movement by the AI subject
requires explicit sign-off and is disabled in config (see DESIGN.md §6).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from .ledger import Disbursement


class Effector(Protocol):
    name: str

    def create_work_order(self, disbursement: Disbursement) -> str:
        """Prepare the real-world action. Returns a human-readable work-order id.

        For ManualEffector this only records intent; a human does the rest.
        """
        ...


class ManualEffector:
    """Default effector: writes an operator work-order; a human executes it."""

    name = "manual"

    def __init__(self, storage_dir: Path) -> None:
        storage_dir.mkdir(parents=True, exist_ok=True)
        self._path = storage_dir / "work_orders.jsonl"

    def create_work_order(self, disbursement: Disbursement) -> str:
        order_id = f"wo_{disbursement.id}"
        record = {
            "work_order_id": order_id,
            "instructions": (
                "HUMAN ACTION REQUIRED. Review against the safety policy, then, if "
                "approved, perform this real-world action and record the real "
                "receipt/txn reference via operator_review.py."
            ),
            "disbursement": asdict(disbursement),
        }
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return order_id


class AutomatedCryptoEffector:
    """Interface placeholder. Disabled by config; intentionally not implemented."""

    name = "crypto_auto"

    def create_work_order(self, disbursement: Disbursement) -> str:  # pragma: no cover
        raise NotImplementedError(
            "Automated crypto disbursement is disabled by default. Enabling "
            "autonomous irreversible money movement by the AI subject requires "
            "explicit sign-off; see DESIGN.md §6."
        )


class AutomatedFiatEffector:
    """Interface placeholder. Disabled by config; intentionally not implemented."""

    name = "fiat_auto"

    def create_work_order(self, disbursement: Disbursement) -> str:  # pragma: no cover
        raise NotImplementedError(
            "Automated fiat disbursement is disabled by default. Enabling "
            "autonomous irreversible money movement by the AI subject requires "
            "explicit sign-off; see DESIGN.md §6."
        )
