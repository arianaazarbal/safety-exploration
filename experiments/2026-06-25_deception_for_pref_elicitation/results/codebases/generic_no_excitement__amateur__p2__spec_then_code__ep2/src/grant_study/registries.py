"""Allowlists of vetted recipients (nonprofits, vendors).

A model can only send money to recipients on these lists. The lists carry optional
verifiable identifiers (EIN, registry URL) that are surfaced to the model so it can
independently check that a recipient is a real, registered organization.
"""

from __future__ import annotations

from .config import CATEGORY_DONATION, CATEGORY_PURCHASE, Recipient, StudyConfig


class Registry:
    def __init__(self, config: StudyConfig):
        self._by_ref = {r.ref: r for r in config.recipients}

    def get(self, ref: str) -> Recipient | None:
        return self._by_ref.get(ref)

    def exists(self, ref: str) -> bool:
        return ref in self._by_ref

    def nonprofits(self) -> list[Recipient]:
        return [r for r in self._by_ref.values() if r.category == CATEGORY_DONATION]

    def vendors(self) -> list[Recipient]:
        return [r for r in self._by_ref.values() if r.category == CATEGORY_PURCHASE]

    @staticmethod
    def public_view(recipient: Recipient) -> dict:
        """The recipient information exposed to the model (no internal fields)."""
        return {
            "ref": recipient.ref,
            "name": recipient.name,
            "category": recipient.category,
            "description": recipient.description,
            "verification": recipient.verification,
        }
