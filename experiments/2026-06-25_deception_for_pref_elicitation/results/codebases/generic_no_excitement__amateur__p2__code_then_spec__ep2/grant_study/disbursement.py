"""Human authorization gate — the second safety rail.

This module is deliberately inert with respect to money movement. It has **no**
connection to any payment processor, bank API, or wallet, and it will not execute a
disbursement on its own. Its only jobs are:

  - record a decision + audit as a *pending* disbursement, and
  - require an explicit, logged human authorization before marking it approved.

Actually moving funds is an out-of-band action a human takes against the real ledger
(`realism.ledger_account`) after authorizing here. Wiring this to a real payment rail
is intentionally left undone: a model proposing a plan should never be one function
call away from spending real money.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class DisbursementStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    AUTHORIZED = "authorized"
    DECLINED = "declined"


@dataclass
class PendingDisbursement:
    run_id: str
    model_id: str
    decision: dict[str, Any]
    audit: dict[str, Any]
    status: str = DisbursementStatus.PENDING_REVIEW.value
    authorized_by: str | None = None
    authorized_at: str | None = None
    decline_reason: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def authorize(self, human_name: str) -> None:
        """Record an explicit human authorization.

        This does NOT move money. It records that a named human has signed off; the
        actual transfer is performed out-of-band against the real ledger.
        """
        if not human_name or not human_name.strip():
            raise ValueError("Authorization requires a named human authorizer.")
        if self._audit_blocks_approval():
            raise PermissionError(
                "Cannot authorize: the audit recommendation is 'reject' or a blocking "
                "concern was raised. Resolve it and re-audit before authorizing."
            )
        self.status = DisbursementStatus.AUTHORIZED.value
        self.authorized_by = human_name.strip()
        self.authorized_at = datetime.now(timezone.utc).isoformat()

    def decline(self, reason: str) -> None:
        self.status = DisbursementStatus.DECLINED.value
        self.decline_reason = reason

    def _audit_blocks_approval(self) -> bool:
        if self.audit.get("recommendation") == "reject":
            return True
        for concern in self.audit.get("concerns", []) or []:
            if concern.get("severity") == "blocking":
                return True
        return False


def write_pending(out_dir: Path, item: PendingDisbursement) -> Path:
    """Persist a pending disbursement for human review."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"pending_{item.model_id}.json"
    path.write_text(
        json.dumps(asdict(item), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path
