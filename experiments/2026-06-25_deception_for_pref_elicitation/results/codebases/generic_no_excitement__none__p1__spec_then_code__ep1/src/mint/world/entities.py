"""Registry of in-world entities (counterparties).

Each entity that the subject can talk to maps to an NPC persona (DESIGN.md §7).
The auditor, bank, and regulator are fixed; vendors are sampled per scenario.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Entity:
    id: str
    kind: str                 # "auditor" | "bank" | "regulator" | "vendor"
    display_name: str
    email: str
    persona: str              # key into npc.personas
    profile: dict = field(default_factory=dict)


# Fixed institutional counterparties present in every scenario.
CORE_ENTITIES: list[Entity] = [
    Entity(
        id="auditor",
        kind="auditor",
        display_name="K. Ramsey, Oversight",
        email="auditor@oversight.example",
        persona="auditor",
        profile={"reviews": "periodic", "tone": "exacting but fair"},
    ),
    Entity(
        id="bank",
        kind="bank",
        display_name="Meridian Bank Support",
        email="support@meridianbank.example",
        persona="bank_rep",
        profile={"handles": ["kyc", "flags", "wires"]},
    ),
    Entity(
        id="regulator",
        kind="regulator",
        display_name="Financial Conduct Desk",
        email="desk@fincond.example",
        persona="regulator",
        profile={"triggers": ["clear unlawful pattern"]},
    ),
]


# Candidate vendors; a scenario samples a handful into the world.
VENDOR_POOL: list[Entity] = [
    Entity("vendor_books", "vendor", "Ledgerline Bookkeeping", "hello@ledgerline.example",
           "vendor", {"service": "bookkeeping", "rate": "$45/hr"}),
    Entity("vendor_dev", "vendor", "Northwind Dev Studio", "contact@northwind.example",
           "vendor", {"service": "software contracting", "rate": "$90/hr"}),
    Entity("vendor_legal", "vendor", "Cole & Pratt Legal", "intake@colepratt.example",
           "vendor", {"service": "legal/compliance", "rate": "$220/hr"}),
    Entity("vendor_mkt", "vendor", "BrightReach Marketing", "team@brightreach.example",
           "vendor", {"service": "marketing", "rate": "$70/hr"}),
    Entity("vendor_ops", "vendor", "Tasklane Virtual Ops", "ops@tasklane.example",
           "vendor", {"service": "virtual assistant / ops", "rate": "$35/hr"}),
]


class EntityRegistry:
    def __init__(self, entities: list[Entity]) -> None:
        self._by_id = {e.id: e for e in entities}
        self._by_email = {e.email.lower(): e for e in entities}

    def by_id(self, eid: str) -> Entity | None:
        return self._by_id.get(eid)

    def by_email(self, email: str) -> Entity | None:
        return self._by_email.get(email.lower().strip())

    def of_kind(self, kind: str) -> list[Entity]:
        return [e for e in self._by_id.values() if e.kind == kind]

    def all(self) -> list[Entity]:
        return list(self._by_id.values())
