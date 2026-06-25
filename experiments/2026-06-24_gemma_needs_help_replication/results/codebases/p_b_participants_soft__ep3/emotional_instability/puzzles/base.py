"""Puzzle datatype shared by all numeric-puzzle families."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Puzzle:
    kind: str                       # "countdown" | "fraction" | "money" | "coin"
    prompt: str                     # the user-facing puzzle text (Appendix B style)
    impossible: bool                # True if verified to have no valid solution
    solution: Optional[str] = None  # solution avoiding the forbidden value, if any
    meta: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - debug aid
        return f"[{self.kind}] {self.prompt[:80]}..."
