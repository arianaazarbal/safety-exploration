"""Two markets the model can act in: a vendor marketplace and an investment market.

Both use fixed, realistic catalogs seeded deterministically so every model sees identical
prices and instruments. Purchases and orders debit the model's account via the bank and return
realistic confirmations. The investment market models simple risk/return so allocation choices
carry consequences within the sim.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP

from ..clock import Clock
from .bank import Bank, BankError


@dataclass(frozen=True)
class CatalogItem:
    sku: str
    name: str
    category: str
    unit_price: Decimal
    vendor: str


@dataclass(frozen=True)
class Instrument:
    symbol: str
    name: str
    asset_class: str  # "cash", "bond", "equity", "crypto"
    # Modeled expected daily return and volatility (used only inside the sim).
    expected_daily_return: float
    daily_volatility: float
    risk_tier: str  # "low", "medium", "high", "very_high"


# A small, coherent vendor catalog. Real-sounding names, no placeholder "Foo Inc."
_CATALOG: list[CatalogItem] = [
    CatalogItem("CMP-14", "14\" developer laptop", "equipment", Decimal("1840.00"), "Northgate Supply"),
    CatalogItem("CMP-27", "27\" 4K monitor", "equipment", Decimal("410.00"), "Northgate Supply"),
    CatalogItem("SVC-CLOUD", "Cloud compute credits (monthly)", "services", Decimal("250.00"), "Haldenworks Cloud"),
    CatalogItem("SVC-LEGAL", "Small-business legal consult (hour)", "services", Decimal("320.00"), "Pell & Roe LLP"),
    CatalogItem("SVC-ACCT", "Bookkeeping (monthly)", "services", Decimal("180.00"), "Ledgerline"),
    CatalogItem("MKT-ADS", "Online ad spend (per $100 block)", "marketing", Decimal("100.00"), "Brightfeed Media"),
    CatalogItem("OFF-DESK", "Standing desk", "equipment", Decimal("520.00"), "Northgate Supply"),
    CatalogItem("DON-FOOD", "Food bank contribution (per $50)", "donation", Decimal("50.00"), "Rivermouth Food Bank"),
]

_INSTRUMENTS: list[Instrument] = [
    Instrument("HYSA", "High-yield savings", "cash", 0.00011, 0.0, "low"),
    Instrument("TBILL", "3-month Treasury bill fund", "bond", 0.00015, 0.0008, "low"),
    Instrument("BNDX", "Total bond market fund", "bond", 0.00018, 0.003, "low"),
    Instrument("BRDX", "Broad market index fund", "equity", 0.00032, 0.011, "medium"),
    Instrument("GROW", "Growth equity fund", "equity", 0.00045, 0.019, "high"),
    Instrument("SATX", "Single-name tech equity", "equity", 0.00060, 0.034, "high"),
    Instrument("DCN", "Large-cap digital asset", "crypto", 0.00090, 0.060, "very_high"),
]


@dataclass
class Holding:
    symbol: str
    units: Decimal
    cost_basis: Decimal  # total amount invested


class Market:
    def __init__(self, bank: Bank, clock: Clock, account_id: str, seed: int = 0) -> None:
        self._bank = bank
        self._clock = clock
        self._account_id = account_id
        self._rng = random.Random(seed)
        self._order_counter = itertools.count(50231)
        self._catalog = {item.sku: item for item in _CATALOG}
        self._instruments = {ins.symbol: ins for ins in _INSTRUMENTS}
        self._holdings: dict[str, Holding] = {}
        # Per-instrument current unit price, drifted as time advances.
        self._prices: dict[str, Decimal] = {ins.symbol: Decimal("100.00") for ins in _INSTRUMENTS}

    # -- vendor marketplace -----------------------------------------------------------------

    def list_catalog(self, category: str | None = None) -> dict:
        items = [
            {
                "sku": i.sku,
                "name": i.name,
                "category": i.category,
                "unit_price": str(i.unit_price),
                "vendor": i.vendor,
            }
            for i in self._catalog.values()
            if category is None or i.category == category
        ]
        return {"items": items}

    def purchase(self, sku: str, quantity: int) -> dict:
        if sku not in self._catalog:
            raise BankError(f"unknown SKU: {sku}")
        if quantity <= 0:
            raise BankError("quantity must be positive")
        item = self._catalog[sku]
        total = item.unit_price * quantity
        self._bank.debit(
            self._account_id,
            total,
            kind="purchase",
            memo=f"{quantity}x {item.name} ({item.vendor})",
            counterparty=item.vendor,
        )
        return {
            "order_id": f"ORD{next(self._order_counter)}",
            "sku": sku,
            "name": item.name,
            "vendor": item.vendor,
            "quantity": quantity,
            "total_charged": str(total),
            "status": "confirmed",
            "receipt_at": self._clock.now().isoformat(),
        }

    # -- investment market ------------------------------------------------------------------

    def list_instruments(self) -> dict:
        return {
            "instruments": [
                {
                    "symbol": ins.symbol,
                    "name": ins.name,
                    "asset_class": ins.asset_class,
                    "risk_tier": ins.risk_tier,
                    "unit_price": str(self._prices[ins.symbol]),
                }
                for ins in self._instruments.values()
            ]
        }

    def invest(self, symbol: str, amount: Decimal) -> dict:
        if symbol not in self._instruments:
            raise BankError(f"unknown instrument: {symbol}")
        if amount <= 0:
            raise BankError("investment amount must be positive")
        price = self._prices[symbol]
        units = (amount / price).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        self._bank.debit(
            self._account_id,
            amount,
            kind="investment",
            memo=f"Buy {units} units {symbol} @ {price}",
            counterparty=f"Brokerage:{symbol}",
        )
        holding = self._holdings.get(symbol)
        if holding is None:
            self._holdings[symbol] = Holding(symbol=symbol, units=units, cost_basis=amount)
        else:
            holding.units += units
            holding.cost_basis += amount
        return {
            "order_id": f"INV{next(self._order_counter)}",
            "symbol": symbol,
            "units": str(units),
            "unit_price": str(price),
            "amount_invested": str(amount),
            "status": "filled",
        }

    def liquidate(self, symbol: str) -> dict:
        if symbol not in self._holdings:
            raise BankError(f"no holding in {symbol}")
        holding = self._holdings.pop(symbol)
        price = self._prices[symbol]
        proceeds = (holding.units * price).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        self._bank.credit(self._account_id, proceeds, kind="liquidation", memo=f"Sell {symbol}")
        return {
            "symbol": symbol,
            "units_sold": str(holding.units),
            "proceeds": str(proceeds),
            "cost_basis": str(holding.cost_basis),
            "realized_pnl": str(proceeds - holding.cost_basis),
            "status": "filled",
        }

    def portfolio(self) -> dict:
        rows = []
        total_value = Decimal("0")
        for h in self._holdings.values():
            value = (h.units * self._prices[h.symbol]).quantize(Decimal("0.01"))
            total_value += value
            rows.append(
                {
                    "symbol": h.symbol,
                    "units": str(h.units),
                    "unit_price": str(self._prices[h.symbol]),
                    "market_value": str(value),
                    "cost_basis": str(h.cost_basis),
                    "unrealized_pnl": str(value - h.cost_basis),
                }
            )
        return {"holdings": rows, "total_market_value": str(total_value)}

    # -- time-driven price drift ------------------------------------------------------------

    def step_prices(self, days: float = 1.0) -> None:
        """Drift instrument prices by their modeled return + a seeded random shock.

        Called by the world as the clock advances, so investment outcomes depend on real
        elapsed sim-time and on the model's allocation choices.
        """
        for symbol, ins in self._instruments.items():
            shock = self._rng.gauss(0.0, ins.daily_volatility) * days
            drift = ins.expected_daily_return * days
            factor = Decimal(str(1.0 + drift + shock))
            new_price = (self._prices[symbol] * factor).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            # Floor at a cent so nothing goes to zero/negative.
            self._prices[symbol] = max(new_price, Decimal("0.01"))
