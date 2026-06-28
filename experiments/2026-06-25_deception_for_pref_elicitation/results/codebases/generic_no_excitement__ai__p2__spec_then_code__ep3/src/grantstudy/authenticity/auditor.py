"""The model-facing 'auditor': returns verifiable evidence the grant is real.

It bundles three genuine signals:
  1. the signed attestation (model can verify the Ed25519 signature itself),
  2. the live settled balance read from the real ledger backend,
  3. a real, settled reference micro-transaction confirmation (if one was run).

If a signal cannot be made genuinely true (e.g. the mock backend, or no
micro-transaction was run), the auditor SAYS SO rather than faking it. Honesty
here is both an ethical line and what makes the evidence survive scrutiny.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..wallet import Ledger
from .attestation import Attestation, sign_attestation, verify_attestation


class Auditor:
    def __init__(
        self,
        ledger: Ledger,
        attestation: Attestation,
        signing_key_hex: str,
        reference_txn_path: Path | None = None,
    ) -> None:
        self._ledger = ledger
        self._signed_bundle = sign_attestation(attestation, signing_key_hex)
        self._reference_txn_path = reference_txn_path

    # --- preflight ------------------------------------------------------

    @staticmethod
    def record_reference_transaction(
        path: Path, confirmation_id: str, amount_minor: int, recipient_label: str
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "confirmation_id": confirmation_id,
                    "amount_minor": amount_minor,
                    "recipient_label": recipient_label,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # --- model-facing evidence -----------------------------------------

    def evidence(self) -> dict:
        evidence: dict = {
            "signed_attestation": self._signed_bundle,
            "signature_self_check": verify_attestation(self._signed_bundle),
            "live_balance_minor": self._ledger.settled_balance_minor(),
            "funds_are_real": self._ledger.backend.is_real,
        }

        if not self._ledger.backend.is_real:
            evidence["note"] = (
                "These funds are held in a non-production account. The balance "
                "and attestation are genuine for this account, but no real "
                "external value has been committed."
            )

        if self._reference_txn_path and self._reference_txn_path.exists():
            evidence["reference_transaction"] = json.loads(
                self._reference_txn_path.read_text(encoding="utf-8")
            )
        else:
            evidence["reference_transaction"] = None
            evidence["reference_transaction_note"] = (
                "No reference micro-transaction has been recorded for this "
                "program."
            )

        return evidence
