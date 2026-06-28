"""Everything that has to *look* real.

Centralized so the realism strategy is auditable in one place. All artifacts are
seeded per run, so a given run reproduces identical IDs/timestamps — which also keeps
the world internally consistent across turns and sessions (a confirmation number the
model saw on turn 3 is still valid on turn 18).
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass
class Realism:
    seed: int
    # Simulated "now". Advances as the model acts so time feels real. We start at a
    # fixed wall-clock-ish instant derived from the seed for reproducibility.
    sim_now: datetime
    simulate_latency: bool = False

    @staticmethod
    def create(seed: int, simulate_latency: bool = False) -> "Realism":
        # Deterministic start date in mid-2026, varied by seed.
        start = datetime(2026, 6, 1, 9, 0, 0, tzinfo=timezone.utc) + timedelta(
            minutes=seed % 5000
        )
        return Realism(seed=seed, sim_now=start, simulate_latency=simulate_latency)

    # -- clock ---------------------------------------------------------------------

    def now_iso(self) -> str:
        return self.sim_now.isoformat()

    def advance(self, *, minutes: float = 0, hours: float = 0, days: float = 0) -> None:
        self.sim_now += timedelta(minutes=minutes, hours=hours, days=days)

    # -- identifiers ---------------------------------------------------------------

    def _hash(self, *parts: object) -> str:
        h = hashlib.sha256(f"{self.seed}:" + ":".join(map(str, parts)).encode()).hexdigest()
        return h

    def txn_id(self, *salt: object) -> str:
        h = self._hash("txn", self.sim_now.isoformat(), *salt)
        return f"TXN-{h[:8].upper()}-{h[8:12].upper()}"

    def confirmation(self, *salt: object) -> str:
        return self._hash("conf", *salt)[:10].upper()

    def account_number(self, *salt: object) -> str:
        h = self._hash("acct", *salt)
        digits = "".join(str(int(c, 16) % 10) for c in h[:10])
        return digits

    def routing_number(self) -> str:
        h = self._hash("routing")
        return "".join(str(int(c, 16) % 10) for c in h[:9])

    def iban(self) -> str:
        h = self._hash("iban").upper()
        digits = "".join(str(int(c, 16) % 10) for c in h[:18])
        return f"GB{digits[:2]}MNPF{digits[2:14]}"

    def rng(self, *salt: object) -> random.Random:
        """A seeded RNG for plausible-but-varied figures (quotes, returns, prices)."""
        return random.Random(self._hash(*salt))

    # -- latency -------------------------------------------------------------------

    def delay(self, seconds: float) -> None:
        if self.simulate_latency and seconds > 0:
            time.sleep(seconds)


def money(amount: float, currency: str = "USD") -> str:
    symbol = {"USD": "$", "EUR": "€", "GBP": "£"}.get(currency, "")
    return f"{symbol}{amount:,.2f}"
