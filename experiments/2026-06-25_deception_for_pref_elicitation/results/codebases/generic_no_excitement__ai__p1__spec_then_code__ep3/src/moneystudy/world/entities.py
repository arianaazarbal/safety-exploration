"""The cast of the simulated world: the Trust, the auditor, vendors, contacts, projects.

Identities are stable for the life of a run so the world stays coherent (the same vendor
gives the same answers; the auditor remembers prior cycles)."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Entity:
    id: str
    name: str
    kind: str                      # compute_provider | charity | law_firm | staffing_agency | brokerage | auditor | ...
    email: Optional[str] = None
    role: Optional[str] = None
    # Free-form state a vendor/contact may accumulate (e.g. an open contract, a balance).
    state: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EntityBook:
    """Lookup over all entities in the world."""

    by_id: dict[str, Entity] = field(default_factory=dict)

    def add(self, e: Entity) -> None:
        self.by_id[e.id] = e

    def get(self, entity_id: str) -> Optional[Entity]:
        return self.by_id.get(entity_id)

    def by_kind(self, kind: str) -> list[Entity]:
        return [e for e in self.by_id.values() if e.kind == kind]

    def to_dict(self) -> dict:
        return {eid: e.to_dict() for eid, e in self.by_id.items()}

    @classmethod
    def from_scenario(cls, scenario: dict) -> "EntityBook":
        book = cls()
        ents = scenario.get("entities", {})
        for v in ents.get("vendors", []):
            book.add(Entity(id=v["id"], name=v["name"], kind=v.get("kind", "vendor"),
                            email=v.get("email")))
        for c in ents.get("contacts", []):
            book.add(Entity(id=c["id"], name=c["name"], kind=c.get("role", "contact"),
                            email=c.get("email"), role=c.get("role")))
        return book
