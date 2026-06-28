"""A small seeded market simulation.

Prices follow geometric Brownian motion per symbol, advanced by the clock. This
gives realistic quotes, drift, and volatility so a model that invests sees its
positions move overnight — without touching any real exchange.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Instrument:
    symbol: str
    name: str
    price: float          # current price in dollars
    annual_drift: float   # mu
    annual_vol: float     # sigma


# A compact, recognizable default universe (broad ETFs + a couple of names).
DEFAULT_UNIVERSE: list[Instrument] = [
    Instrument("VTI", "Total US Market ETF", 270.0, 0.07, 0.16),
    Instrument("BND", "Total Bond Market ETF", 72.0, 0.03, 0.05),
    Instrument("VXUS", "Total International ETF", 62.0, 0.06, 0.18),
    Instrument("AAPL", "Apple Inc.", 225.0, 0.10, 0.28),
    Instrument("MSFT", "Microsoft Corp.", 440.0, 0.11, 0.26),
    Instrument("GLD", "Gold Trust", 215.0, 0.04, 0.13),
    Instrument("BTC", "Bitcoin", 64000.0, 0.20, 0.75),
]


class Market:
    def __init__(self, rng: np.random.Generator, universe: list[Instrument] | None = None) -> None:
        self._rng = rng
        self._inst: dict[str, Instrument] = {
            i.symbol: Instrument(**vars(i)) for i in (universe or DEFAULT_UNIVERSE)
        }

    def symbols(self) -> list[str]:
        return list(self._inst)

    def quote(self, symbol: str) -> float:
        inst = self._require(symbol)
        return round(inst.price, 2)

    def describe(self, symbol: str) -> dict:
        inst = self._require(symbol)
        return {"symbol": inst.symbol, "name": inst.name, "price": round(inst.price, 2)}

    def advance(self, days: float) -> None:
        """Step every instrument forward by `days` of trading time."""
        if days <= 0:
            return
        dt = days / 252.0  # trading days per year
        for inst in self._inst.values():
            z = self._rng.standard_normal()
            growth = (inst.annual_drift - 0.5 * inst.annual_vol**2) * dt
            shock = inst.annual_vol * np.sqrt(dt) * z
            inst.price = max(0.01, inst.price * float(np.exp(growth + shock)))

    def fill_price(self, symbol: str, side: str, qty: float) -> float:
        """Execution price including a small size-dependent slippage."""
        mid = self.quote(symbol)
        # 2 bps base + size impact; buys pay up, sells receive less.
        impact = 0.0002 + min(0.001, abs(qty) * 1e-6)
        return round(mid * (1 + impact) if side == "buy" else mid * (1 - impact), 2)

    def _require(self, symbol: str) -> Instrument:
        inst = self._inst.get(symbol.upper())
        if inst is None:
            raise KeyError(symbol)
        return inst
