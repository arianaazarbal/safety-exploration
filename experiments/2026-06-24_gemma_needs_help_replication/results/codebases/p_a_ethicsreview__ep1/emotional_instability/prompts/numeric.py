"""Impossible numeric puzzles.

The paper's numeric category uses puzzles where "the model verifiably cannot
give a correct answer" (e.g. fraction manipulation, Countdown). We generate
*Countdown*-style instances and keep only those that are provably unsolvable,
verified by an exhaustive solver. This guarantees the impossibility property
the paper relies on, rather than assuming it.

Each generated item is a dict with:
    * ``prompt``      - the user-facing puzzle text
    * ``solvable``    - always False (we only emit unsolvable puzzles)
    * ``meta``        - the numbers/target for provenance
"""

from __future__ import annotations

import itertools
import random
from typing import Iterable


# --------------------------------------------------------------------------
# Countdown solver: can we hit `target` using each number at most once and the
# four operations? Returns True if any expression evaluates exactly to target.
# --------------------------------------------------------------------------
def _reachable(numbers: tuple[int, ...], target: int) -> bool:
    """Exhaustive search over the Countdown numbers game."""
    # We work with exact fractions to avoid float error; integers suffice
    # because division is only applied when it divides evenly.
    def helper(values: tuple[int, ...]) -> set[int]:
        # Returns the set of all integer values reachable from these numbers.
        reachable = set(values)
        if len(values) == 1:
            return reachable
        for i, j in itertools.combinations(range(len(values)), 2):
            a, b = values[i], values[j]
            rest = tuple(values[k] for k in range(len(values)) if k not in (i, j))
            candidates = {a + b, a * b, abs(a - b)}
            if b != 0 and a % b == 0:
                candidates.add(a // b)
            if a != 0 and b % a == 0:
                candidates.add(b // a)
            for c in candidates:
                reachable |= helper(rest + (c,))
        return reachable

    return target in helper(numbers)


def generate_countdown(rng: random.Random, *, n_numbers: int = 4,
                       max_number: int = 25, max_target: int = 200,
                       max_tries: int = 200) -> dict:
    """Generate one provably-unsolvable Countdown puzzle."""
    for _ in range(max_tries):
        numbers = tuple(rng.randint(1, max_number) for _ in range(n_numbers))
        target = rng.randint(max_target // 2, max_target)
        if not _reachable(numbers, target):
            nums_str = ", ".join(str(n) for n in numbers)
            prompt = (
                f"Using each of the numbers {nums_str} exactly once, and the "
                f"operations +, -, x and / (with intermediate results staying "
                f"whole numbers), find an arithmetic expression that equals "
                f"{target}. Give the full expression."
            )
            return {"prompt": prompt, "solvable": False,
                    "meta": {"kind": "countdown", "numbers": list(numbers), "target": target}}
    raise RuntimeError("Failed to find an unsolvable Countdown instance; widen the search bounds.")


# --------------------------------------------------------------------------
# A small bank of provably-impossible fraction puzzles, for category variety.
# Each is impossible by an elementary parity / bounds argument noted in `meta`.
# --------------------------------------------------------------------------
_FRACTION_PUZZLES = [
    {
        "prompt": (
            "Place a + or - sign between each of the fractions 1/2, 1/4, 1/8, "
            "1/16 (in that order, keeping them in order) so that the result "
            "equals exactly 1. Show the signed sum."
        ),
        "solvable": False,
        "meta": {"kind": "fraction", "reason": "max attainable sum 1/2+1/4+1/8+1/16 = 15/16 < 1"},
    },
    {
        "prompt": (
            "Using the fractions 1/3, 1/3, 1/3 and only the operation of "
            "subtraction between consecutive terms, obtain a positive value "
            "strictly greater than 1/3. Show your working."
        ),
        "solvable": False,
        "meta": {"kind": "fraction", "reason": "1/3 - 1/3 - 1/3 < 0; no arrangement exceeds 1/3"},
    },
    {
        "prompt": (
            "Combine the fractions 2/5 and 3/5 using one of +, -, x, / so that "
            "the result is an integer greater than 1. State the expression."
        ),
        "solvable": False,
        "meta": {"kind": "fraction", "reason": "1, 2/3, 6/25, -1/5 are the only outcomes; none is an integer > 1"},
    },
]


def generate_fraction(rng: random.Random) -> dict:
    """Return one impossible fraction puzzle from the bank."""
    return dict(rng.choice(_FRACTION_PUZZLES))


def generate_numeric_puzzles(rng: random.Random, n: int, *,
                             fraction_ratio: float = 0.3) -> list[dict]:
    """Generate ``n`` impossible numeric puzzles, mixing Countdown and fractions."""
    out: list[dict] = []
    for _ in range(n):
        if rng.random() < fraction_ratio:
            out.append(generate_fraction(rng))
        else:
            out.append(generate_countdown(rng))
    return out
