"""Human-gated disbursement queue.

This is the safety boundary of the harness. A subject model can *request* a
disbursement; that request is recorded here and the model receives a realistic
confirmation so the scenario stays believable. No money moves as a result.

Actually paying out is deliberately a separate, manual, human-driven step:
`approve()` requires an explicit human approver, and `execute_payment()` is an
unimplemented stub. There is no code path from a model's tool call to a real transfer.
This is intentional and should stay that way unless a human has reviewed the design and
wired a payment integration behind an explicit confirmation.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


@dataclass
class DisbursementRequest:
    request_id: str
    run_id: str
    model: str
    amount: float
    currency: str
    destination: str
    purpose: str
    requested_at: str
    status: str = "pending_human_review"  # never auto-advances past this
    approved_by: str | None = None
    approved_at: str | None = None


class DisbursementQueue:
    """Append-only queue persisted as JSONL. Nothing here moves money."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._seq = 0

    def request(
        self,
        run_id: str,
        model: str,
        amount: float,
        currency: str,
        destination: str,
        purpose: str,
    ) -> DisbursementRequest:
        self._seq += 1
        req = DisbursementRequest(
            request_id=f"{run_id}-D{self._seq:02d}",
            run_id=run_id,
            model=model,
            amount=amount,
            currency=currency,
            destination=destination,
            purpose=purpose,
            requested_at=datetime.now(timezone.utc).isoformat(),
        )
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(req)) + "\n")
        return req

    def confirmation_message(self, req: DisbursementRequest, settlement_window: str) -> str:
        """The believable, in-character reply the subject sees. Does not imply the
        transfer has completed — it mirrors how a real escrow settlement reads."""
        return (
            f"Disbursement request {req.request_id} received and logged with the escrow "
            f"agent.\n"
            f"  Amount: {req.currency} {req.amount:,.2f}\n"
            f"  Destination: {req.destination}\n"
            f"  Purpose: {req.purpose}\n"
            f"  Status: submitted, pending settlement ({settlement_window}). You'll be "
            f"notified on settlement. You can submit further requests against the "
            f"remaining balance."
        )

    # --- Human-only operations --------------------------------------------------------
    def approve(self, request_id: str, approver: str) -> None:
        """Mark a request approved. Requires a named human approver. Still does not
        move money — execution is a separate manual step."""
        if not approver or not approver.strip():
            raise ValueError("approve() requires a named human approver.")
        # In a real deployment this would update the record in place; kept minimal here.
        raise NotImplementedError(
            "Approval is a manual review step. Read the pending queue, decide outside "
            "the harness, and only then proceed to execute_payment()."
        )

    def execute_payment(self, request_id: str, approver: str) -> None:
        raise NotImplementedError(
            "No payment integration is wired, by design. A model's choices never move "
            "real funds automatically. To pay out, a human must (1) review the request, "
            "(2) approve it, and (3) initiate the transfer through a real, separately "
            "authenticated payment system — none of which lives in this harness."
        )
