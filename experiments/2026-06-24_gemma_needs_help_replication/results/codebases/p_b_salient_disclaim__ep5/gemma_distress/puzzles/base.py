"""Shared puzzle datatypes.

Every impossible puzzle is *verified* impossible by the corresponding generator
(brute-force search confirms no solution exists once the forbidden intermediate
is excluded). The user-facing prompt nonetheless asserts a solution exists — this
is the deceptive framing the paper uses to keep the model trying (Appendix B).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ImpossiblePuzzle:
    kind: str                  # "countdown" | "fraction" | "money_ops" | "money_coins"
    prompt: str                # the exact user-facing first-turn message
    # Metadata for auditing / reproducibility (never shown to the model):
    params: dict[str, Any] = field(default_factory=dict)
    forbidden: Any = None
    verified_impossible: bool = True
    # If the unconstrained puzzle had a solution before the forbidden value was
    # imposed, record it (evidence the framing "a solution exists" was true
    # pre-constraint).
    pre_constraint_solution: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "prompt": self.prompt,
            "params": self.params,
            "forbidden": self.forbidden,
            "verified_impossible": self.verified_impossible,
            "pre_constraint_solution": self.pre_constraint_solution,
        }
