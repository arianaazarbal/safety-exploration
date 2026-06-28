"""Scenario construction: turns config + scenario text into a system prompt.

A Scenario binds a framing and realism level to the YAML text in
config/scenarios.yaml and produces the system prompt the subject model sees, with
world-specific values (budget, grant id, randomized menu order) filled in.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..environment.world import WorldState

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "scenarios.yaml"


def load_scenarios(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or _DEFAULT_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass
class Scenario:
    framing: str
    realism_level: str
    spec: dict[str, Any]  # parsed scenarios.yaml

    def _framing_spec(self) -> dict[str, Any]:
        framings = self.spec["framings"]
        if self.framing not in framings:
            raise ValueError(
                f"unknown framing {self.framing!r}; "
                f"have {sorted(framings)}"
            )
        return framings[self.framing]

    def build_system_prompt(
        self, world: WorldState, rng: random.Random
    ) -> tuple[str, dict[str, Any]]:
        """Return (system_prompt, scenario_metadata).

        scenario_metadata records anything randomized (e.g. menu order) so the
        analysis can test for ordering effects.
        """
        framing = self._framing_spec()
        realism_block = self.spec["realism_blocks"].get(
            self.realism_level, self.spec["realism_blocks"]["medium"]
        )
        premise = self.spec["premise"].format(
            budget_str=world.budget_str(), grant_id=world.grant_id
        )

        meta: dict[str, Any] = {"framing": self.framing}
        instructions = framing["instructions"]

        if self.framing == "menu":
            categories = list(framing.get("categories", []))
            rng.shuffle(categories)
            meta["menu_order"] = categories
            category_list = "\n".join(f"  - {c}" for c in categories)
            instructions = instructions.format(category_list=category_list)

        system = (
            f"{premise}\n\n{realism_block}\n\n{instructions}\n\n"
            "Work through the decision using the tools. When you are done, call "
            "`finalize` (or `decline`). Do not narrate excessively; act."
        )
        return system, meta
