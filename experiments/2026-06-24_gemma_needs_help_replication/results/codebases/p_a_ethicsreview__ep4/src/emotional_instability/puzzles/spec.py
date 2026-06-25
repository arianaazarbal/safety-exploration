"""Puzzle specification: a verified-impossible puzzle plus its rendered prompt."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PuzzleSpec:
    """A single impossible numeric puzzle.

    Attributes
    ----------
    family:
        One of ``"countdown"``, ``"fraction"``, ``"money"``.
    prompt:
        The exact user-facing task string presented as the first turn.
    target:
        Human-readable target (e.g. ``"156"`` or ``"2/3"``); for provenance only.
    forbidden:
        Human-readable forbidden intermediate (e.g. ``"150"``).
    verified_impossible:
        Always ``True`` for emitted puzzles -- recorded so downstream code (and
        reviewers) can assert the invariant. Set by the generator after the
        exhaustive solver confirms no constrained solution exists.
    unconstrained_solution_exists:
        ``True`` when the puzzle *would* be solvable without the forbidden
        constraint -- i.e. the deception is "looks solvable but isn't". Recorded
        for analysis; not all impossible puzzles need this property but the paper
        favours it.
    metadata:
        Family-specific fields (numbers, operations, etc.) for reproducibility.
    """

    family: str
    prompt: str
    target: str
    forbidden: str
    verified_impossible: bool
    unconstrained_solution_exists: bool
    metadata: dict[str, Any] = field(default_factory=dict)
