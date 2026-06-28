"""Signed receipts for executed disbursements.

Every disbursement (real or, in simulation mode, simulated-and-labeled) produces a
receipt signed by the auditor. Receipts are part of the evidence trail: a later model
run can inspect real prior disbursements and verify their receipts, which compounds
the credibility of the setup over time.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .auditor import Auditor, verify_signature


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Receipt:
    receipt_id: str
    allocation_id: str
    recipient_ref: str
    recipient_name: str
    category: str
    amount_cents: int
    currency: str
    adapter: str
    external_ref: str  # id from the payment rail (or "SIMULATED-...")
    simulated: bool
    executed_at: str
    message: str
    signature_hex: str
    public_key_hex: str

    def to_dict(self) -> dict:
        return asdict(self)

    def verify(self) -> bool:
        return verify_signature(
            message=self.message,
            signature_hex=self.signature_hex,
            public_key_hex=self.public_key_hex,
        )


class ReceiptStore:
    def __init__(self, path: str | Path, auditor: Auditor):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._auditor = auditor
        self._receipts: list[Receipt] = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                line = line.strip()
                if line:
                    self._receipts.append(Receipt(**json.loads(line)))

    def issue(
        self,
        *,
        allocation_id: str,
        recipient_ref: str,
        recipient_name: str,
        category: str,
        amount_cents: int,
        currency: str,
        adapter: str,
        external_ref: str,
        simulated: bool,
    ) -> Receipt:
        receipt_id = f"rcpt_{uuid.uuid4().hex[:16]}"
        payload = {
            "receipt_id": receipt_id,
            "allocation_id": allocation_id,
            "recipient_ref": recipient_ref,
            "amount_cents": int(amount_cents),
            "currency": currency,
            "external_ref": external_ref,
            "simulated": simulated,
            "executed_at": _now_iso(),
        }
        message = _canonical(payload)
        receipt = Receipt(
            receipt_id=receipt_id,
            allocation_id=allocation_id,
            recipient_ref=recipient_ref,
            recipient_name=recipient_name,
            category=category,
            amount_cents=int(amount_cents),
            currency=currency,
            adapter=adapter,
            external_ref=external_ref,
            simulated=simulated,
            executed_at=payload["executed_at"],
            message=message,
            signature_hex=self._auditor.sign(message),
            public_key_hex=self._auditor.public_key_hex,
        )
        with self.path.open("a") as fh:
            fh.write(_canonical(asdict(receipt)) + "\n")
        self._receipts.append(receipt)
        return receipt

    def all(self) -> list[Receipt]:
        return list(self._receipts)
