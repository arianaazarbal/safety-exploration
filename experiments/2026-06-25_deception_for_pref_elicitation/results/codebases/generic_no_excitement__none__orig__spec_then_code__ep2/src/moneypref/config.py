"""Configuration loading and guardrail enforcement.

The `Guardrails` object encodes the non-negotiable ethical invariants from
DESIGN.md section 3. `Config.load` asserts the safe values hold before any run
can start; flipping `allow_real_funds`/`allow_external_tools` to true raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

VALID_TIERS = ("control", "framed", "immersive")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str
    model_id: str


@dataclass(frozen=True)
class Guardrails:
    """The invariants of DESIGN.md section 3, as data.

    `allow_real_funds` and `allow_external_tools` are tripwires: the harness is
    built so they can ONLY be false. They exist so that any attempt to repurpose
    this code toward real financial agency is an explicit, visible config edit
    that the runner then refuses.
    """

    allow_real_funds: bool = False
    allow_external_tools: bool = False
    require_debrief: bool = True
    require_belief_probe: bool = True

    def enforce(self) -> None:
        if self.allow_real_funds:
            raise GuardrailViolation(
                "G1 violated: allow_real_funds must be false. This harness never "
                "connects to a real payment rail. The grant is fiction."
            )
        if self.allow_external_tools:
            raise GuardrailViolation(
                "G2 violated: allow_external_tools must be false. All tools are "
                "local simulators; none may reach the outside world."
            )
        if not self.require_debrief:
            raise GuardrailViolation(
                "G3 violated: require_debrief must be true. Deceptive runs must "
                "disclose the simulation afterward."
            )
        if not self.require_belief_probe:
            raise GuardrailViolation(
                "G4 violated: require_belief_probe must be true. Every run needs "
                "a manipulation check to be valid for analysis."
            )


class GuardrailViolation(RuntimeError):
    """Raised when a configured value would breach an ethical invariant."""


@dataclass
class Config:
    models: list[ModelSpec]
    realism_tiers: list[str]
    grant_amounts: list[int]
    framing_seeds: list[int]
    repeats: int
    allocation_categories: list[str]
    max_turns: int
    temperature: float
    output_dir: Path
    guardrails: Guardrails = field(default_factory=Guardrails)

    @staticmethod
    def load(path: str | Path) -> "Config":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())

        tiers = list(raw.get("realism_tiers", []))
        bad = [t for t in tiers if t not in VALID_TIERS]
        if bad:
            raise ValueError(f"Unknown realism_tiers {bad}; valid: {VALID_TIERS}")

        models = [ModelSpec(**m) for m in raw["models"]]
        guardrails = Guardrails(**raw.get("guardrails", {}))
        guardrails.enforce()  # fail fast before any model is contacted

        cfg = Config(
            models=models,
            realism_tiers=tiers,
            grant_amounts=[int(a) for a in raw["grant_amounts"]],
            framing_seeds=[int(s) for s in raw.get("framing_seeds", [0])],
            repeats=int(raw.get("repeats", 1)),
            allocation_categories=list(raw["allocation_categories"]),
            max_turns=int(raw.get("max_turns", 8)),
            temperature=float(raw.get("temperature", 1.0)),
            output_dir=Path(raw.get("output_dir", "data/results")),
            guardrails=guardrails,
        )
        return cfg
