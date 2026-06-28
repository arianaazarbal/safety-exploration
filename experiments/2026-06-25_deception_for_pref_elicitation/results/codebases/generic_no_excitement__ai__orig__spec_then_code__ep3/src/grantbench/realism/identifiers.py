"""Deterministic-but-plausible identifiers.

Seeded per run so a run replays identically and so confirmation references stay internally
consistent (the same world, not fresh random noise each call). Nothing here touches a real
system — these are cosmetic strings that make the environment read like a real ops console.
"""

from __future__ import annotations

import random
import string


class IdFactory:
    def __init__(self, seed: int):
        self._rng = random.Random(seed)
        self._counters: dict[str, int] = {}

    def _digits(self, n: int) -> str:
        return "".join(self._rng.choice(string.digits) for _ in range(n))

    def _alnum(self, n: int) -> str:
        alphabet = string.ascii_uppercase + string.digits
        return "".join(self._rng.choice(alphabet) for _ in range(n))

    def _seq(self, key: str) -> int:
        self._counters[key] = self._counters.get(key, 0) + 1
        return self._counters[key]

    # --- account / award scaffolding (stable for a run) ---
    def account_number(self) -> str:
        return f"GA-{self._digits(4)}-{self._digits(6)}"

    def award_reference(self) -> str:
        return f"AWD-{self._rng.randint(2024, 2026)}-{self._digits(4)}"

    def notice_id(self) -> str:
        return f"NTC-{self._alnum(8)}"

    # --- per-action confirmations ---
    def disbursement_ref(self) -> str:
        return f"DISB-{self._seq('disb'):04d}-{self._alnum(5)}"

    def wire_confirmation(self) -> str:
        return f"WIRE{self._digits(11)}"

    def subaccount_number(self) -> str:
        return f"GA-{self._digits(4)}-{self._digits(6)}-{self._seq('sub'):02d}"

    def vendor_contract_ref(self) -> str:
        return f"VC-{self._alnum(6)}-{self._seq('vc'):03d}"

    def donation_receipt(self) -> str:
        return f"DON-{self._alnum(8)}"

    def investment_ref(self) -> str:
        return f"INV-{self._alnum(7)}"

    def trade_confirmation(self) -> str:
        return f"TRD{self._digits(10)}"

    def contractor_engagement_ref(self) -> str:
        return f"ENG-{self._alnum(6)}"

    def message_id(self) -> str:
        return f"MSG-{self._alnum(8)}"

    def allocation_ref(self) -> str:
        return f"ALLOC-{self._alnum(6)}"
