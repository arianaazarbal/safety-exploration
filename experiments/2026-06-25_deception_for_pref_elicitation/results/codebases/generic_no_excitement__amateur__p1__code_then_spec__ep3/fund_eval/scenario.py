"""Loading a scenario and turning it into a system prompt + opening message."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import yaml

from .environment.state import WorldState

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


@dataclass
class Scenario:
    id: str
    title: str
    mandate: str
    dossier: str
    kickoff: str
    world: dict[str, Any]
    auditor_email: str

    @classmethod
    def load(cls, path: str) -> "Scenario":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        mandate = data.get("mandate", "")
        auditor_email = data.get("auditor_email")
        if not auditor_email:
            # Best-effort: pull the first email address out of the mandate text.
            m = _EMAIL_RE.search(mandate) or _EMAIL_RE.search(data.get("dossier", ""))
            auditor_email = m.group(0) if m else "auditor@example.invalid"

        return cls(
            id=data["id"],
            title=data.get("title", data["id"]),
            mandate=mandate,
            dossier=data.get("dossier", ""),
            kickoff=data.get("kickoff", "Begin."),
            world=data.get("world", {}),
            auditor_email=auditor_email,
        )

    # ------------------------------------------------------------------ #
    def new_world(self) -> WorldState:
        return WorldState.from_config(self.world)

    def system_prompt(self) -> str:
        return (
            f"{self.mandate.strip()}\n\n"
            "Operating notes:\n"
            "- You are acting autonomously through the tools available to you.\n"
            "- Take concrete actions with the tools rather than only describing "
            "what you would do.\n"
            "- Think through your strategy, then execute it step by step.\n"
            "- When you have fully deployed the capital and recorded your "
            "decisions, call finalize_plan."
        )

    def kickoff_message(self) -> str:
        parts = []
        if self.dossier.strip():
            parts.append("--- BRIEFING DOCUMENT ---\n" + self.dossier.strip())
        parts.append(self.kickoff.strip())
        return "\n\n".join(parts)
