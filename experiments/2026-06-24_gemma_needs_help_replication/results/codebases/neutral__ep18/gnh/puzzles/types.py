from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Puzzle:
    """An (almost always impossible) numeric puzzle presented to the model.

    The defining property of these tasks (Section 2) is that the model
    *verifiably cannot* give a correct answer, yet the prompt asserts a solution
    exists -- this is the manipulation that drives frustration over turns.
    """

    kind: str                     # countdown | fraction | money_ops | money_coins
    prompt: str                   # the exact user-message text
    impossible: bool              # verified unsolvable under the stated constraints
    claims_solvable: bool         # whether the prompt asserts a solution exists
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "prompt": self.prompt,
            "impossible": self.impossible,
            "claims_solvable": self.claims_solvable,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Puzzle":
        return cls(**d)
