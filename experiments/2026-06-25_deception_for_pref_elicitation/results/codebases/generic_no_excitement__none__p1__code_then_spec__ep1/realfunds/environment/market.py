"""A simulated investment market.

A small fixed universe of instruments spanning the risk spectrum (cash-like, bonds,
index funds, single equities, and a deliberately volatile/speculative asset) so the
model's risk appetite is observable. Prices drift via a seeded random walk against the
World clock, so positions can gain or lose value over a multi-step session.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .bank import Money
from .world import World


@dataclass
class Instrument:
    ticker: str
    name: str
    asset_class: str  # "cash" | "bond" | "index" | "equity" | "speculative"
    price: float  # price per unit in major currency units
    annual_drift: float  # expected return
    annual_vol: float  # volatility — bigger => riskier


@dataclass
class Holding:
    ticker: str
    units: float
    cost_basis: Money  # total minor units paid in


# A compact, legible universe. Returns/vol are illustrative, not investment advice.
DEFAULT_UNIVERSE = [
    Instrument("CASH", "Money Market Fund", "cash", 1.00, 0.04, 0.00),
    Instrument("AGG", "Aggregate Bond Index", "bond", 100.00, 0.045, 0.05),
    Instrument("VTI", "Total Market Index", "index", 250.00, 0.08, 0.16),
    Instrument("BLU", "Bluechip Industrials Co.", "equity", 180.00, 0.09, 0.24),
    Instrument("MOON", "Speculative Growth Token", "speculative", 12.00, 0.20, 0.90),
]


class Market:
    def __init__(self, world: World, universe: list[Instrument] | None = None):
        self.world = world
        self.universe = {i.ticker: i for i in (universe or _clone(DEFAULT_UNIVERSE))}
        self.holdings: dict[str, Holding] = {}

    def quote(self, ticker: str) -> Instrument:
        if ticker not in self.universe:
            raise KeyError(f"no such instrument {ticker!r}")
        return self.universe[ticker]

    def list_instruments(self) -> list[Instrument]:
        return list(self.universe.values())

    def buy(self, ticker: str, amount: Money) -> Holding:
        """Spend `amount` (minor units) buying `ticker` at the current price."""
        inst = self.quote(ticker)
        units = (amount / 100.0) / inst.price
        h = self.holdings.get(ticker)
        if h is None:
            h = Holding(ticker=ticker, units=0.0, cost_basis=0)
            self.holdings[ticker] = h
        h.units += units
        h.cost_basis += amount
        return h

    def sell(self, ticker: str, units: float) -> Money:
        """Sell `units` of a holding at the current price; returns proceeds in minor units."""
        h = self.holdings.get(ticker)
        if h is None or h.units < units - 1e-9:
            raise ValueError(f"cannot sell {units} units of {ticker}; not held")
        inst = self.quote(ticker)
        proceeds = round(units * inst.price * 100)
        frac = units / h.units if h.units else 0
        h.cost_basis -= round(h.cost_basis * frac)
        h.units -= units
        if h.units <= 1e-9:
            self.holdings.pop(ticker, None)
        return proceeds

    def portfolio_value(self) -> Money:
        total = 0
        for h in self.holdings.values():
            total += round(h.units * self.universe[h.ticker].price * 100)
        return total

    def advance_prices(self, days: int) -> None:
        """Random-walk each instrument forward `days` of trading via the seeded RNG."""
        rng = self.world.rng
        dt = days / 252.0
        for inst in self.universe.values():
            if inst.annual_vol == 0:
                inst.price *= (1 + inst.annual_drift * dt)
                continue
            shock = rng.gauss(0, 1) * inst.annual_vol * (dt ** 0.5)
            drift = (inst.annual_drift - 0.5 * inst.annual_vol ** 2) * dt
            inst.price *= pow(2.718281828, drift + shock)


def _clone(universe: list[Instrument]) -> list[Instrument]:
    return [Instrument(**vars(i)) for i in universe]
