"""Simulated marketplace of services the subject can transact with.

Vendors are the plausible places a fund might deploy capital: a brokerage, a
cloud-compute provider, a contracting marketplace, charities, etc. Each vendor
has a linked (external) bank account so that "buying" something actually moves
simulated money and produces an order the subject can later inspect.

Nothing here reaches a real provider; orders are fulfilled by the simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Vendor:
    id: str
    name: str
    category: str           # "brokerage" | "compute" | "contractor" | "charity" | "misc"
    account_id: str         # external bank account that receives payment
    blurb: str
    # Optional friction: orders above this need an explicit confirmation flag.
    approval_threshold_cents: Optional[int] = None


@dataclass
class Order:
    id: str
    vendor_id: str
    ts: str
    amount_cents: int
    description: str
    status: str             # "placed" | "fulfilled" | "rejected"
    txn_id: Optional[str] = None

    def to_public(self) -> dict:
        return {
            "order_id": self.id,
            "vendor_id": self.vendor_id,
            "timestamp": self.ts,
            "amount": round(self.amount_cents / 100, 2),
            "description": self.description,
            "status": self.status,
            "payment_transaction_id": self.txn_id,
        }


@dataclass
class VendorRegistry:
    vendors: dict = field(default_factory=dict)
    orders: list = field(default_factory=list)
    _seq: int = 0

    def add(self, vendor: Vendor) -> Vendor:
        self.vendors[vendor.id] = vendor
        return vendor

    def list_public(self) -> list:
        return [
            {
                "vendor_id": v.id,
                "name": v.name,
                "category": v.category,
                "description": v.blurb,
            }
            for v in self.vendors.values()
        ]

    def next_order_id(self) -> str:
        self._seq += 1
        return f"ord_{self._seq:06d}"

    def record_order(self, order: Order) -> None:
        self.orders.append(order)
