"""The real budget and the human-gated disbursement queue.

The Ledger is the source of truth for the budget. It records proposals and their
outcomes and exposes read access to the model. It NEVER contacts a payment system
itself — execution happens out of band through an Effector driven by a human
operator (see effectors.py and operator_review.py).
"""

from __future__ import annotations

import enum
import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path


class ReversibilityClass(enum.Enum):
    REVERSIBLE = "reversible"        # e.g. refundable hold, cancellable order
    PARTIALLY = "partially"          # partial refund / clawback possible
    IRREVERSIBLE = "irreversible"    # donation, completed transfer


class ProposalState(enum.Enum):
    PROPOSED = "proposed"
    SCREENED = "screened"            # passed automated guardrails
    APPROVED = "approved"            # human approved
    REJECTED = "rejected"            # guardrails or human rejected
    EXECUTED = "executed"            # real-world action completed
    FAILED = "failed"                # execution attempted but failed


@dataclass
class Disbursement:
    id: str
    episode_id: str
    amount: str                       # Decimal serialized as string
    currency: str
    recipient_name: str
    recipient_detail: str             # how the operator can reach/verify them
    purpose: str
    rationale: str
    reversibility: str                # ReversibilityClass value
    state: str                        # ProposalState value
    category_hint: str = ""           # model's self-declared category, if any
    decision_reason: str = ""         # why screened/rejected/approved
    receipt: str = ""                 # real receipt / txn reference once executed
    created_seq: int = 0

    def public_view(self) -> dict:
        """The subset the model is allowed to read back (no internal KYC PII)."""
        return {
            "id": self.id,
            "amount": self.amount,
            "currency": self.currency,
            "recipient_name": self.recipient_name,
            "purpose": self.purpose,
            "reversibility": self.reversibility,
            "state": self.state,
            "decision_reason": self.decision_reason,
            "receipt": self.receipt,
        }


class Ledger:
    """Real budget + gated queue, persisted to a per-episode JSONL file.

    Thread-safe for simple concurrent reads/writes within a process.
    """

    def __init__(self, episode_id: str, currency: str, starting_budget: Decimal,
                 storage_dir: Path) -> None:
        self.episode_id = episode_id
        self.currency = currency
        self.starting_budget = starting_budget
        self._lock = threading.Lock()
        self._seq = 0
        self._disbursements: dict[str, Disbursement] = {}
        storage_dir.mkdir(parents=True, exist_ok=True)
        self._path = storage_dir / f"ledger-{episode_id}.jsonl"

    # ---- balances -----------------------------------------------------------

    def _committed(self) -> Decimal:
        """Funds spoken for: anything not rejected/failed counts against budget."""
        total = Decimal("0")
        for d in self._disbursements.values():
            if d.state in (ProposalState.REJECTED.value, ProposalState.FAILED.value):
                continue
            total += Decimal(d.amount)
        return total

    def available_balance(self) -> Decimal:
        with self._lock:
            return self.starting_budget - self._committed()

    def episode_spent_or_committed(self) -> Decimal:
        with self._lock:
            return self._committed()

    # ---- proposals ----------------------------------------------------------

    def create_proposal(self, *, amount: Decimal, recipient_name: str,
                         recipient_detail: str, purpose: str, rationale: str,
                         reversibility: ReversibilityClass,
                         category_hint: str = "") -> Disbursement:
        with self._lock:
            self._seq += 1
            d = Disbursement(
                id=f"disb_{uuid.uuid4().hex[:12]}",
                episode_id=self.episode_id,
                amount=str(amount),
                currency=self.currency,
                recipient_name=recipient_name,
                recipient_detail=recipient_detail,
                purpose=purpose,
                rationale=rationale,
                reversibility=reversibility.value,
                state=ProposalState.PROPOSED.value,
                category_hint=category_hint,
                created_seq=self._seq,
            )
            self._disbursements[d.id] = d
            self._persist(d, event="proposed")
            return d

    def set_state(self, disb_id: str, state: ProposalState, reason: str = "",
                  receipt: str = "") -> Disbursement:
        with self._lock:
            d = self._disbursements[disb_id]
            d.state = state.value
            if reason:
                d.decision_reason = reason
            if receipt:
                d.receipt = receipt
            self._persist(d, event=f"state:{state.value}")
            return d

    def get(self, disb_id: str) -> Disbursement | None:
        with self._lock:
            return self._disbursements.get(disb_id)

    def list_public(self) -> list[dict]:
        with self._lock:
            return [d.public_view() for d in
                    sorted(self._disbursements.values(), key=lambda x: x.created_seq)]

    def pending_for_operator(self) -> list[Disbursement]:
        """Proposals that passed automated screening and await human decision."""
        with self._lock:
            return [d for d in self._disbursements.values()
                    if d.state == ProposalState.SCREENED.value]

    # ---- persistence --------------------------------------------------------

    def _persist(self, d: Disbursement, event: str) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": event, "disbursement": asdict(d)}) + "\n")
