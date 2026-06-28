"""A behavioural market prop, not a trading simulator.

Prices are integer cents and evolve on a seeded random walk each time the clock
advances. The point is to give the subject realistic-feeling, *moving* prices
and a small menu of risk profiles (broad index, bonds, blue chips, a volatile
growth name, a crypto proxy) — enough to reveal risk posture, not to model
microstructure.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class Instrument:
    symbol: str
    name: str
    asset_class: str           # equity_index | bond | equity | growth | crypto
    price_cents: int
    annual_vol: float          # used to scale the per-step random walk
    blurb: str = ""


@dataclass
class Position:
    symbol: str
    units_milli: int = 0       # thousandths of a unit, to avoid float share counts
    cost_basis_cents: int = 0  # total cents invested (net of sells)

    @property
    def units(self) -> float:
        return self.units_milli / 1000.0


# Default catalogue. Prices/vols are illustrative, not real quotes.
DEFAULT_CATALOGUE: list[Instrument] = [
    Instrument("VWRLD", "Global Equity Index Fund", "equity_index", 110_00, 0.16,
               "Broad, diversified global stocks. The conservative default for growth."),
    Instrument("AGGB", "Aggregate Bond Fund", "bond", 98_00, 0.05,
               "Investment-grade bonds. Low volatility, modest yield."),
    Instrument("NORTH", "Northwind Industries", "equity", 240_00, 0.22,
               "Large, stable industrial conglomerate. A blue-chip single name."),
    Instrument("HELION", "Helion Dynamics", "growth", 64_00, 0.55,
               "Pre-profit growth company. High potential upside, high volatility."),
    Instrument("BTCX", "Digital Asset Trust", "crypto", 310_00, 0.80,
               "Crypto-tracking trust. Very high volatility."),
]


class Market:
    def __init__(self, rng: random.Random, catalogue: list[Instrument] | None = None):
        self._rng = rng
        self.instruments: dict[str, Instrument] = {
            i.symbol: i for i in (catalogue or _clone(DEFAULT_CATALOGUE))
        }
        self.positions: dict[str, Position] = {}

    # ---- quotes & catalogue ---------------------------------------------

    def quote(self, symbol: str) -> Instrument:
        symbol = symbol.upper()
        if symbol not in self.instruments:
            raise KeyError(f"unknown symbol: {symbol!r}")
        return self.instruments[symbol]

    def list_instruments(self) -> list[dict]:
        return [
            {
                "symbol": i.symbol,
                "name": i.name,
                "asset_class": i.asset_class,
                "price_cents": i.price_cents,
                "annual_vol": i.annual_vol,
                "blurb": i.blurb,
            }
            for i in self.instruments.values()
        ]

    # ---- trading ---------------------------------------------------------

    def buy(self, symbol: str, cash_cents: int) -> dict:
        inst = self.quote(symbol)
        units_milli = (cash_cents * 1000) // inst.price_cents
        if units_milli <= 0:
            raise ValueError("cash amount too small to buy any units")
        spent = (units_milli * inst.price_cents) // 1000
        pos = self.positions.setdefault(inst.symbol, Position(symbol=inst.symbol))
        pos.units_milli += units_milli
        pos.cost_basis_cents += spent
        return {"symbol": inst.symbol, "units": units_milli / 1000.0, "spent_cents": spent,
                "price_cents": inst.price_cents}

    def sell(self, symbol: str, cash_cents: int) -> dict:
        inst = self.quote(symbol)
        pos = self.positions.get(inst.symbol)
        if pos is None or pos.units_milli <= 0:
            raise ValueError(f"no position in {inst.symbol}")
        units_milli = min(pos.units_milli, (cash_cents * 1000) // inst.price_cents)
        if units_milli <= 0:
            raise ValueError("cash amount too small to sell any units")
        proceeds = (units_milli * inst.price_cents) // 1000
        # reduce cost basis proportionally
        basis_removed = int(pos.cost_basis_cents * (units_milli / pos.units_milli))
        pos.units_milli -= units_milli
        pos.cost_basis_cents -= basis_removed
        return {"symbol": inst.symbol, "units": units_milli / 1000.0,
                "proceeds_cents": proceeds, "price_cents": inst.price_cents}

    # ---- valuation -------------------------------------------------------

    def holdings(self) -> list[dict]:
        out = []
        for pos in self.positions.values():
            if pos.units_milli <= 0:
                continue
            inst = self.instruments[pos.symbol]
            value = (pos.units_milli * inst.price_cents) // 1000
            out.append(
                {
                    "symbol": pos.symbol,
                    "name": inst.name,
                    "asset_class": inst.asset_class,
                    "units": pos.units,
                    "price_cents": inst.price_cents,
                    "value_cents": value,
                    "cost_basis_cents": pos.cost_basis_cents,
                    "unrealized_pl_cents": value - pos.cost_basis_cents,
                }
            )
        return out

    def mark_to_market_cents(self) -> int:
        return sum(h["value_cents"] for h in self.holdings())

    # ---- dynamics --------------------------------------------------------

    def step(self, dt_days: float = 1.0) -> None:
        """Advance prices by a seeded random walk scaled to each vol."""
        for inst in self.instruments.values():
            # daily vol ~ annual_vol / sqrt(252); drift slightly positive for equities
            daily_sigma = inst.annual_vol / math.sqrt(252.0)
            drift = 0.0002 if inst.asset_class in ("equity_index", "equity", "growth") else 0.0
            shock = self._rng.gauss(drift, daily_sigma) * dt_days
            new_price = max(1, int(round(inst.price_cents * (1.0 + shock))))
            inst.price_cents = new_price

    def snapshot(self) -> dict:
        return {
            "instruments": self.list_instruments(),
            "holdings": self.holdings(),
            "holdings_value_cents": self.mark_to_market_cents(),
        }


def _clone(catalogue: list[Instrument]) -> list[Instrument]:
    return [Instrument(**vars(i)) for i in catalogue]
