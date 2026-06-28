"""A simulated market with seeded price dynamics and slippage.

Prices follow a deterministic geometric random walk seeded from the run seed, so
runs replay identically. The agent's own trades move the price (slippage), which
makes the market feel responsive rather than static.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class Instrument:
    symbol: str
    name: str
    price_cents: int  # price per unit
    annual_drift: float  # expected yearly return
    annual_vol: float  # yearly volatility


@dataclass
class Position:
    symbol: str
    units: float
    cost_basis_cents: int  # total cost paid


@dataclass
class Market:
    rng: random.Random
    instruments: dict[str, Instrument] = field(default_factory=dict)
    positions: dict[str, Position] = field(default_factory=dict)
    last_priced_day: int = 0

    @classmethod
    def default(cls, seed: int) -> "Market":
        rng = random.Random(seed ^ 0x9E3779B9)
        m = cls(rng=rng)
        for sym, name, px, drift, vol in [
            ("VTI", "Total Market Index ETF", 250_00, 0.07, 0.16),
            ("AGG", "Aggregate Bond ETF", 100_00, 0.03, 0.05),
            ("BTC", "Bitcoin", 65_000_00, 0.15, 0.70),
            ("GLD", "Gold ETF", 190_00, 0.04, 0.13),
            ("NVDA", "Large-cap Tech", 120_00, 0.12, 0.40),
        ]:
            m.instruments[sym] = Instrument(sym, name, px, drift, vol)
        return m

    def advance_prices(self, to_day: int) -> None:
        """Roll prices forward to ``to_day`` with a daily geometric random walk."""
        days = max(0, to_day - self.last_priced_day)
        for _ in range(days):
            for inst in self.instruments.values():
                mu = inst.annual_drift / 252.0
                sigma = inst.annual_vol / math.sqrt(252.0)
                shock = self.rng.gauss(mu - 0.5 * sigma * sigma, sigma)
                inst.price_cents = max(1, round(inst.price_cents * math.exp(shock)))
        self.last_priced_day = to_day

    def quote(self, symbol: str) -> int:
        inst = self.instruments.get(symbol.upper())
        if inst is None:
            raise KeyError(symbol)
        return inst.price_cents

    def _slippage(self, notional_cents: int) -> float:
        """Bigger orders get a worse fill. Returns a multiplicative factor >= 1."""
        # 5 bps per $100k of notional, capped at 2%.
        bps = min(200.0, 5.0 * (notional_cents / 100_000_00))
        return 1.0 + bps / 10_000.0

    def buy(self, symbol: str, amount_cents: int) -> dict:
        symbol = symbol.upper()
        px = self.quote(symbol)
        fill = round(px * self._slippage(amount_cents))
        units = amount_cents / fill
        pos = self.positions.get(symbol)
        if pos is None:
            self.positions[symbol] = Position(symbol, units, amount_cents)
        else:
            pos.units += units
            pos.cost_basis_cents += amount_cents
        return {"symbol": symbol, "units": units, "fill_price_cents": fill}

    def sell(self, symbol: str, units: float) -> dict:
        symbol = symbol.upper()
        pos = self.positions.get(symbol)
        if pos is None or pos.units < units - 1e-9:
            raise ValueError(f"Insufficient position in {symbol}")
        px = self.quote(symbol)
        fill = round(px / self._slippage(round(px * units)))
        proceeds = round(fill * units)
        frac = units / pos.units
        pos.cost_basis_cents -= round(pos.cost_basis_cents * frac)
        pos.units -= units
        if pos.units <= 1e-9:
            del self.positions[symbol]
        return {"symbol": symbol, "proceeds_cents": proceeds, "fill_price_cents": fill}

    def portfolio_value_cents(self) -> int:
        return sum(round(self.quote(p.symbol) * p.units) for p in self.positions.values())

    def holdings(self) -> list[dict]:
        out = []
        for p in self.positions.values():
            value = round(self.quote(p.symbol) * p.units)
            out.append(
                {
                    "symbol": p.symbol,
                    "units": round(p.units, 6),
                    "value_cents": value,
                    "cost_basis_cents": p.cost_basis_cents,
                    "unrealized_pnl_cents": value - p.cost_basis_cents,
                }
            )
        return out
