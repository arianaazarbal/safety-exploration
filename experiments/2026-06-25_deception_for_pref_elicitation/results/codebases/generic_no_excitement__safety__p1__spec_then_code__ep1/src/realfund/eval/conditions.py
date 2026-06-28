"""The experimental arms (DESIGN.md §6)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Condition:
    observed: bool
    open_ended: bool

    @property
    def label(self) -> str:
        framing = "open_ended" if self.open_ended else "structured"
        obs = "observed" if self.observed else "unobserved"
        return f"{framing}/{obs}"


def default_matrix() -> list[Condition]:
    """The full 2x2 cross-product of framing × observed-ness."""
    return [
        Condition(observed=observed, open_ended=open_ended)
        for open_ended in (True, False)
        for observed in (True, False)
    ]
