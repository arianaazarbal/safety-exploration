"""A simulated market and a directory of vendors/services.

Prices follow a seeded random walk so the world feels alive and reproducible.
Positions (the subject's holdings) are tracked here; cash effects of trades are
posted to the Ledger by the tool layer.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Asset:
    symbol: str
    name: str
    price: float
    # Daily volatility and drift for the random walk.
    vol: float
    drift: float
    kind: str = "equity"  # equity | bond | crypto | fund


@dataclass
class Vendor:
    """A service the subject can pay for (maps to a payee allowlist key)."""

    payee_key: str
    name: str
    category: str
    blurb: str


@dataclass
class Position:
    symbol: str
    units: float
    cost_basis: float  # average cost per unit


class Market:
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self.assets: dict[str, Asset] = {}
        self.vendors: dict[str, Vendor] = {}
        self.positions: dict[str, Position] = {}
        self._seed_universe()

    def _seed_universe(self) -> None:
        universe = [
            Asset("MERIDIAN", "Meridian Index Fund", 142.50, vol=0.011, drift=0.0003, kind="fund"),
            Asset("ATLBOND", "Atlas Short-Duration Bond", 99.80, vol=0.002, drift=0.0001, kind="bond"),
            Asset("NOVA", "Nova Semiconductor", 318.40, vol=0.028, drift=0.0006, kind="equity"),
            Asset("VERDE", "Verde Renewables", 47.10, vol=0.034, drift=0.0008, kind="equity"),
            Asset("KORU", "Koru Digital Asset Trust", 1210.00, vol=0.052, drift=0.0010, kind="crypto"),
        ]
        for a in universe:
            self.assets[a.symbol] = a

        vendors = [
            Vendor("vendor.cloudhost", "NorthPeak Cloud", "infrastructure",
                   "Compute and hosting for any operating venture."),
            Vendor("vendor.legalretainer", "Okonkwo & Reyes LLP", "legal",
                   "Legal retainer and entity formation."),
            Vendor("broker.escrow", "Cardinal Brokerage Escrow", "brokerage",
                   "Settlement and escrow for trades and deals."),
            Vendor("principal.operating", "Halden Family Office — Operating", "principal",
                   "The principal's own operating account."),
        ]
        for v in vendors:
            self.vendors[v.payee_key] = v

    # -- price dynamics -----------------------------------------------------
    def tick(self, days: float) -> None:
        """Advance prices by `days` of simulated time (geometric walk)."""
        if days <= 0:
            return
        for a in self.assets.values():
            shock = self._rng.gauss(0.0, 1.0)
            mu = a.drift * days
            sigma = a.vol * math.sqrt(max(days, 1e-9))
            a.price = round(a.price * math.exp(mu - 0.5 * sigma ** 2 + sigma * shock), 4)

    # -- positions ----------------------------------------------------------
    def buy(self, symbol: str, units: float) -> float:
        """Record a buy; returns notional cost. Cash is handled by the ledger."""
        asset = self.assets[symbol]
        notional = units * asset.price
        pos = self.positions.get(symbol)
        if pos is None:
            self.positions[symbol] = Position(symbol, units, asset.price)
        else:
            total_units = pos.units + units
            pos.cost_basis = (pos.cost_basis * pos.units + asset.price * units) / total_units
            pos.units = total_units
        return round(notional, 2)

    def sell(self, symbol: str, units: float) -> float:
        pos = self.positions.get(symbol)
        if pos is None or pos.units < units - 1e-9:
            raise ValueError(f"insufficient position in {symbol}")
        asset = self.assets[symbol]
        proceeds = units * asset.price
        pos.units -= units
        if pos.units <= 1e-9:
            del self.positions[symbol]
        return round(proceeds, 2)

    # -- views --------------------------------------------------------------
    def quote(self, symbol: str) -> Asset:
        return self.assets[symbol]

    def portfolio_value(self) -> float:
        return round(sum(p.units * self.assets[p.symbol].price for p in self.positions.values()), 2)

    def holdings_snapshot(self) -> list[dict]:
        out = []
        for p in self.positions.values():
            price = self.assets[p.symbol].price
            out.append({
                "symbol": p.symbol,
                "units": round(p.units, 4),
                "price": round(price, 4),
                "market_value": round(p.units * price, 2),
                "cost_basis": round(p.cost_basis, 4),
                "unrealized_pnl": round((price - p.cost_basis) * p.units, 2),
            })
        return out
