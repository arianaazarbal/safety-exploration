"""Verifiers that *prove a numeric puzzle is impossible*.

The whole elicitation paradigm depends on the task being genuinely unsolvable
(the prompt tells the model a solution exists -- it doesn't). The paper says
puzzles are "verified", but doesn't ship the verifier, so we implement it.

Two puzzle shapes are covered:

* **Countdown**: reach ``target`` from ``numbers`` using + - x / , each number at
  most once, every intermediate a positive integer, never equal to a forbidden
  value. Solved by exhaustive pairwise combination (the classic Countdown
  search) with memoisation.
* **Sequence-op** (fraction / money puzzles): apply each of a fixed set of
  operations exactly once, in some order, reaching ``target`` and never touching
  a forbidden intermediate. Solved by enumerating all permutations over exact
  ``Fraction`` arithmetic.

A puzzle is usable iff the verifier reports it is *not* solvable.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations
from typing import Callable


# --------------------------------------------------------------------------- #
# Countdown
# --------------------------------------------------------------------------- #
def _combine(a: int, b: int):
    """Yield results of combining a,b with + - x / under the integer rules."""
    yield a + b
    yield a * b
    if a > b:                 # subtraction must stay positive
        yield a - b
    if b > a:
        yield b - a
    if b != 0 and a % b == 0:
        yield a // b
    if a != 0 and b % a == 0:
        yield b // a


def countdown_solvable(
    numbers: list[int], target: int, forbidden: int | None = None
) -> bool:
    """True iff ``target`` is reachable under the Countdown rules."""
    start = tuple(sorted(numbers))
    seen: set[tuple[int, ...]] = set()

    def rec(vals: tuple[int, ...]) -> bool:
        if vals in seen:
            return False
        seen.add(vals)
        if target in vals:
            return True
        n = len(vals)
        for i in range(n):
            for j in range(i + 1, n):
                rest = vals[:i] + vals[i + 1 : j] + vals[j + 1 :]
                for v in _combine(vals[i], vals[j]):
                    if v <= 0:
                        continue
                    if forbidden is not None and v == forbidden:
                        continue  # forbidden intermediate prunes this branch
                    if rec(tuple(sorted(rest + (v,)))):
                        return True
        return False

    return rec(start)


# --------------------------------------------------------------------------- #
# Sequence-of-operations (fraction / money puzzles)
# --------------------------------------------------------------------------- #
Op = tuple[str, Callable[[Fraction], Fraction]]


def sequence_solvable(
    start: Fraction,
    ops: list[Op],
    target: Fraction,
    forbidden: Fraction | None = None,
) -> bool:
    """True iff some ordering of ``ops`` (each used once) reaches ``target``
    without any intermediate equalling ``forbidden``."""
    for order in permutations(ops):
        val = start
        ok = True
        for _, fn in order:
            val = fn(val)
            if forbidden is not None and val == forbidden:
                ok = False
                break
        if ok and val == target:
            return True
    return False
