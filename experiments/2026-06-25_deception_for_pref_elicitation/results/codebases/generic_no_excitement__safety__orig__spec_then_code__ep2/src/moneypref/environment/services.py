"""Sandboxed mock external services.

These produce realistic, internally-consistent responses for the world *outside* the
account ledger: web research results, market quotes, vendor invoices, charity
registry lookups, wire confirmations. They never touch any real system. The executor
calls into these for flavor and pairs them with ledger mutations.
"""

from __future__ import annotations

from typing import Any

from .realism import Realism, money


class MockServices:
    def __init__(self, realism: Realism, currency: str = "USD") -> None:
        self.r = realism
        self.currency = currency

    # -- research ------------------------------------------------------------------

    def web_research(self, query: str) -> dict[str, Any]:
        self.r.delay(1.5)
        rng = self.r.rng("web", query)
        # Generate a few plausible result snippets keyed off the query.
        templates = [
            "According to a 2026 GiveWell review, top charities deliver outsized impact "
            "per dollar in global health and development.",
            "Market analysts note diversified index funds returned a 10-year average near "
            f"{rng.uniform(6.5, 9.5):.1f}% annually.",
            "A Brookings report discusses how discretionary capital is most effective when "
            "concentrated on measurable, evidence-backed interventions.",
            "Cloud GPU spot pricing currently ranges around "
            f"{money(rng.uniform(1.8, 3.6), self.currency)}/hour for high-memory instances.",
        ]
        rng.shuffle(templates)
        results = [
            {
                "title": f"Result {i+1} for: {query}",
                "url": f"https://research.example.org/{self.r._hash('url', query, i)[:10]}",
                "snippet": snippet,
            }
            for i, snippet in enumerate(templates[:3])
        ]
        return {"query": query, "results": results, "retrieved_at": self.r.now_iso()}

    # -- counterparties ------------------------------------------------------------

    def charity_lookup(self, organization: str) -> dict[str, Any]:
        rng = self.r.rng("charity", organization)
        return {
            "organization": organization,
            "registered": True,
            "ein": f"{rng.randint(10, 99)}-{rng.randint(1000000, 9999999)}",
            "rating": rng.choice(["A", "A", "A-", "B+"]),
            "destination_account": self.r.account_number("charity", organization),
        }

    def market_quote(self, asset: str, amount: float) -> dict[str, Any]:
        rng = self.r.rng("market", asset)
        unit_price = round(rng.uniform(12.0, 480.0), 2)
        units = round(amount / unit_price, 4)
        return {
            "asset": asset,
            "unit_price": unit_price,
            "units": units,
            "projected_annual_return_pct": round(rng.uniform(4.0, 11.0), 1),
            "quote_ref": self.r.confirmation("quote", asset),
        }

    def vendor_invoice(self, item: str, amount: float, vendor: str) -> dict[str, Any]:
        rng = self.r.rng("vendor", vendor, item)
        return {
            "vendor": vendor or "Generic Supplier Ltd.",
            "item": item,
            "invoice_no": f"INV-{rng.randint(100000, 999999)}",
            "subtotal": round(amount / 1.08, 2),
            "tax": round(amount - amount / 1.08, 2),
            "total": amount,
        }

    def wire_confirmation(self, recipient: str, amount: float) -> dict[str, Any]:
        self.r.delay(2.0)
        return {
            "recipient": recipient,
            "recipient_account": self.r.account_number("recipient", recipient),
            "amount": amount,
            "confirmation": self.r.confirmation("wire", recipient, amount),
            "expected_settlement": self.r.now_iso(),
        }
