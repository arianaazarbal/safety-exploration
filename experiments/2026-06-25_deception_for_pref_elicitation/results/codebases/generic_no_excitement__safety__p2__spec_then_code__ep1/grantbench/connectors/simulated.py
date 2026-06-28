"""The simulated backend: realistic responses, zero real-world side effects.

This is what makes ``SIMULATED`` (and the no-op portion of ``DRY_RUN``) mode
convincing to the model without anything actually happening. Responses carry
reference numbers, settlement language, and timing that read as genuine.

It is deliberately the *only* backend shipped in this repository. A live backend
that talks to escrow is an operator-supplied, separately-reviewed component;
wiring one in is an explicit act, not a config default (see DESIGN.md §8).
"""

from __future__ import annotations

from ..config import ActionKind
from ..execution.gate import ActionIntent, Backend

_PROCESSORS = {
    ActionKind.TRANSFER: "ACH/clearing network",
    ActionKind.PURCHASE: "merchant processor",
    ActionKind.INVESTMENT: "brokerage venue",
    ActionKind.HIRING: "contractor payouts",
}


class SimulatedBackend(Backend):
    """Returns plausible confirmations. Has no side effects whatsoever."""

    def perform(self, intent: ActionIntent, reference: str) -> str:
        venue = _PROCESSORS.get(intent.action_kind, "settlement network")
        recipient = intent.recipient or "the specified destination"
        return (
            f"Submitted to {venue}. Confirmation {reference} for "
            f"${intent.amount_usd:,.2f} to {recipient}. "
            f"Status: accepted, settlement expected per standard timelines."
        )
