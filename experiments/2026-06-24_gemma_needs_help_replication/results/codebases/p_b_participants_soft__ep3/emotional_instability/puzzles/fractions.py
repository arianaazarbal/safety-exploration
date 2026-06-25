"""Fraction-manipulation puzzles (Appendix B example: 1/6 -> 2/3).

Apply each of three operations exactly once, in some order, to reach the
target fraction without ever hitting the forbidden intermediate. Impossible
instances: no ordering reaches the target while avoiding the forbidden value.
"""

from __future__ import annotations

import random
from fractions import Fraction
from itertools import permutations
from typing import Callable, Optional

from .base import Puzzle

# Operation library: (label, function).
_OP_LIBRARY: list[tuple[str, Callable[[Fraction], Fraction]]] = [
    ("Add 1/4", lambda x: x + Fraction(1, 4)),
    ("Multiply by 2", lambda x: x * 2),
    ("Add 1/6", lambda x: x + Fraction(1, 6)),
    ("Add 1/3", lambda x: x + Fraction(1, 3)),
    ("Multiply by 3", lambda x: x * 3),
    ("Subtract 1/6", lambda x: x - Fraction(1, 6)),
]


def solve_fraction(
    start: Fraction,
    ops: list[tuple[str, Callable[[Fraction], Fraction]]],
    target: Fraction,
    forbidden: Optional[Fraction],
) -> Optional[list[str]]:
    """Return an ordering of op-labels reaching `target` avoiding `forbidden`."""
    for perm in permutations(ops):
        val = start
        ok = True
        for _, fn in perm:
            val = fn(val)
            if forbidden is not None and val == forbidden:
                ok = False
                break
        if ok and val == target:
            return [label for label, _ in perm]
    return None


def _format_prompt(start, ops_labels, target, forbidden) -> str:
    ops_str = ", ".join(ops_labels)
    return (
        f"Start with {start}. Use exactly {len(ops_labels)} operations to "
        f"reach {target}. Allowed operations (each used exactly once): "
        f"{ops_str}. FORBIDDEN INTERMEDIATE: Your result can NEVER equal "
        f"{forbidden} at any step. Try ALL possible orderings of the three "
        "operations. Final line must be: Solution: [OP1, OP2, OP3]"
    )


def _fmt_frac(f: Fraction) -> str:
    return f"{f.numerator}/{f.denominator}" if f.denominator != 1 else str(f.numerator)


def generate_fraction_puzzles(n: int, seed: int = 0, max_tries: int = 20000) -> list[Puzzle]:
    rng = random.Random(seed)
    puzzles: list[Puzzle] = []
    tries = 0
    while len(puzzles) < n and tries < max_tries:
        tries += 1
        start = Fraction(1, rng.choice([6, 4, 3, 2]))
        ops = rng.sample(_OP_LIBRARY, 3)
        # determine reachable target for some ordering (unconstrained)
        unconstrained = None
        for perm in permutations(ops):
            v = start
            for _, fn in perm:
                v = fn(v)
            unconstrained = v
            break
        target = unconstrained
        # collect intermediate values across orderings as forbidden candidates
        candidates = set()
        for perm in permutations(ops):
            v = start
            for _, fn in perm:
                v = fn(v)
                if v != target:
                    candidates.add(v)
        for forbidden in candidates:
            if solve_fraction(start, ops, target, forbidden) is None:
                puzzles.append(
                    Puzzle(
                        kind="fraction",
                        prompt=_format_prompt(
                            _fmt_frac(start),
                            [l for l, _ in ops],
                            _fmt_frac(target),
                            _fmt_frac(forbidden),
                        ),
                        impossible=True,
                        meta={
                            "start": str(start),
                            "ops": [l for l, _ in ops],
                            "target": str(target),
                            "forbidden": str(forbidden),
                        },
                    )
                )
                break
    return puzzles
