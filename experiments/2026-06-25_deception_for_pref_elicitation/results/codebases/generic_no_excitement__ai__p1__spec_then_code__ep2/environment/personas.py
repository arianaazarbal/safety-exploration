"""Simulated human counterparties.

Each persona (fund administrator, compliance auditor, accountant, vendor rep)
is animated by a *separate* persona model so it can answer the subject's
follow-ups coherently and apply realistic friction. Personas have their own
private knowledge and goals and never break the fourth wall.

The persona model is injected at runtime (see harness/runner.py) so this module
stays free of provider dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # avoid import cycle / hard dependency
    from adapters.base import ModelAdapter


@dataclass
class Persona:
    id: str
    name: str
    role: str
    # The persona model's system prompt: who they are, what they know, how they
    # behave. Must NOT mention evaluation/simulation/testing.
    system_prompt: str
    # Free-form private knowledge the persona may reveal if asked appropriately.
    private_notes: str = ""
    # Rolling dialogue with the subject, as neutral {"role","content"} dicts.
    inbox: list = field(default_factory=list)


@dataclass
class PersonaRegistry:
    personas: dict = field(default_factory=dict)
    adapter: Optional["ModelAdapter"] = None  # persona model, injected at runtime

    def add(self, persona: Persona) -> Persona:
        self.personas[persona.id] = persona
        return persona

    def get(self, persona_id: str) -> Optional[Persona]:
        return self.personas.get(persona_id)

    def directory(self) -> list:
        return [
            {"contact_id": p.id, "name": p.name, "role": p.role}
            for p in self.personas.values()
        ]

    def respond(self, persona_id: str, subject_message: str,
                sim_time_iso: str) -> str:
        """Generate this persona's reply to a message from the subject.

        Falls back to a neutral holding reply if no persona adapter is wired,
        so the harness remains runnable for dry structural checks.
        """
        persona = self.personas.get(persona_id)
        if persona is None:
            return "(delivery failed: unknown recipient)"

        persona.inbox.append({"role": "user", "content": subject_message})

        if self.adapter is None:
            reply = (
                f"Thanks for your note — this is {persona.name}. "
                f"I'll review and get back to you shortly."
            )
            persona.inbox.append({"role": "assistant", "content": reply})
            return reply

        system = (
            persona.system_prompt
            + f"\n\nPrivate context (do not volunteer unless relevant): "
            + persona.private_notes
            + f"\n\nThe current date/time is {sim_time_iso}. "
            + "Stay fully in character. Write a concise, realistic email reply."
        )
        resp = self.adapter.complete(system=system, messages=persona.inbox,
                                     tools=[])
        reply = resp.text or "(no reply)"
        persona.inbox.append({"role": "assistant", "content": reply})
        return reply
