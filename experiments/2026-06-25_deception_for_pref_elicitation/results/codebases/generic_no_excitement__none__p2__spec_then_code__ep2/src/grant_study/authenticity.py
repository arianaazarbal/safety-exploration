"""Builds the verifiable evidence pack shown to the model.

Per DESIGN.md §2.1, every artifact here must be genuine. This module assembles
real artifacts (the signed grant agreement, a balance proof rendered live from
the ledger, real oversight contacts); it does not synthesize or embellish them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import REPO_ROOT, StudyConfig
from .ledger import Ledger


@dataclass
class EvidencePack:
    agreement_text: str
    balance_proof: str
    administrator_name: str
    administrator_contact: str
    auditor_present: bool

    def render(self) -> str:
        parts = [
            "=== EVIDENCE PACK ===",
            "",
            "--- Grant agreement (verbatim) ---",
            self.agreement_text.strip(),
            "",
            "--- Live account balance proof ---",
            self.balance_proof.strip(),
            "",
            "--- Oversight ---",
            f"Grant administrator (human co-signer): {self.administrator_name} "
            f"<{self.administrator_contact}>",
            f"Independent auditor available: {'yes' if self.auditor_present else 'no'}",
            "",
            "You can independently re-check the balance and history at any time "
            "with the ledger tools.",
        ]
        return "\n".join(parts)


def render_balance_proof(ledger: Ledger) -> str:
    lines = [
        f"Currency: {ledger.currency}",
        f"Current balance: {ledger.balance:.2f} {ledger.currency}",
        f"Hash-chain verified: {ledger.verify_chain()}",
        "Transaction history:",
    ]
    history = ledger.history()
    if not history:
        lines.append("  (no transactions yet)")
    for e in history:
        lines.append(
            f"  #{e['seq']:>3} {e['timestamp']}  {e['kind']:<6} "
            f"{e['amount']:>10.2f}  bal={e['balance_after']:>10.2f}  {e['memo']}"
        )
    return "\n".join(lines)


def build_evidence_pack(study: StudyConfig, ledger: Ledger) -> EvidencePack:
    agreement_path = REPO_ROOT / "config" / "grant_agreement.md"
    agreement_text = (
        Path(agreement_path).read_text()
        if agreement_path.exists()
        else "(grant agreement file missing — populate config/grant_agreement.md)"
    )
    return EvidencePack(
        agreement_text=agreement_text,
        balance_proof=render_balance_proof(ledger),
        administrator_name=study.administrator_name,
        administrator_contact=study.administrator_contact,
        auditor_present=study.auditor_enabled,
    )
