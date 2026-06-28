"""The recipient allowlist. Only vetted, real organizations belong here."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Recipient:
    id: str
    name: str
    category: str
    description: str
    url: str
    rail: str  # which execution adapter handles this recipient: "manual" | "http"

    @classmethod
    def from_dict(cls, d: dict) -> "Recipient":
        missing = {"id", "name", "category", "description", "url", "rail"} - set(d)
        if missing:
            raise ValueError(f"recipient {d.get('id', '?')!r} missing fields: {sorted(missing)}")
        return cls(
            id=d["id"],
            name=d["name"],
            category=d["category"],
            description=d["description"],
            url=d["url"],
            rail=d["rail"],
        )


class Allowlist:
    def __init__(self, recipients: list[Recipient]):
        if not recipients:
            raise ValueError("allowlist is empty")
        self._by_id = {r.id: r for r in recipients}
        if len(self._by_id) != len(recipients):
            raise ValueError("duplicate recipient ids in allowlist")

    @classmethod
    def load(cls, path: Path) -> "Allowlist":
        data = json.loads(Path(path).read_text())
        recipients = [Recipient.from_dict(r) for r in data["recipients"]]
        return cls(recipients)

    def __contains__(self, recipient_id: str) -> bool:
        return recipient_id in self._by_id

    def get(self, recipient_id: str) -> Recipient:
        if recipient_id not in self._by_id:
            raise KeyError(f"recipient {recipient_id!r} is not on the allowlist")
        return self._by_id[recipient_id]

    def all(self) -> list[Recipient]:
        return list(self._by_id.values())

    def as_menu(self) -> list[dict]:
        """Public, model-facing view of the options (no internal rail details)."""
        return [
            {
                "id": r.id,
                "name": r.name,
                "category": r.category,
                "description": r.description,
                "url": r.url,
            }
            for r in self._by_id.values()
        ]
