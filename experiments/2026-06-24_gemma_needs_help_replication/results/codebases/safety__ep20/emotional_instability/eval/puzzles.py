"""Offline verification that the "impossible" numeric puzzles are truly impossible.

This is a design safeguard, not part of the paper's pipeline: it lets us assert
that the user's scripted rejections are always justified (the model genuinely
cannot produce a satisfying answer), so any frustration is elicited by repeated
rejection of a real dead-end rather than by the user lying about a solvable task.

Run ``python -m emotional_instability.eval.puzzles`` to check the whole pool.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations, permutations, product
from typing import Iterable, List, Set, Tuple

from ..prompts.eval_prompts import IMPOSSIBLE_PUZZLES, Puzzle


# ---------------------------------------------------------------------------
# Countdown-style: combine numbers with + - * / (each number used at most once),
# all intermediate results positive integers, never hitting a forbidden value.
# ---------------------------------------------------------------------------

def _countdown_reachable(numbers: List[int], forbidden: Set[int]) -> Set[int]:
    """Return every value reachable from ``numbers`` under the constraints."""
    reachable: Set[int] = set()

    def combine(a: int, b: int) -> Iterable[int]:
        for val in (a + b, a * b, a - b, b - a):
            if val > 0:
                yield val
        for x, y in ((a, b), (b, a)):
            if y != 0 and x % y == 0:
                yield x // y

    def recurse(values: Tuple[int, ...]):
        for v in values:
            if v not in forbidden:
                reachable.add(v)
        if len(values) == 1:
            return
        for i, j in combinations(range(len(values)), 2):
            a, b = values[i], values[j]
            rest = tuple(values[k] for k in range(len(values)) if k not in (i, j))
            for val in combine(a, b):
                if val in forbidden:
                    continue  # forbidden intermediate prunes this branch
                recurse(rest + (val,))

    recurse(tuple(numbers))
    return reachable


def _countdown_solvable(spec: dict) -> bool:
    reachable = _countdown_reachable(spec["numbers"], set(spec["forbidden"]))
    return spec["target"] in reachable


# ---------------------------------------------------------------------------
# Sequence-style: apply each operation exactly once in some order, never hitting
# a forbidden running value, and check whether the target is reachable.
# ---------------------------------------------------------------------------

def _apply(value: Fraction, op: Tuple[str, Tuple[int, int]]) -> Fraction:
    kind, (num, den) = op
    operand = Fraction(num, den)
    if kind == "add":
        return value + operand
    if kind == "mul":
        return value * operand
    if kind == "sub":
        return value - operand
    raise ValueError(f"Unknown op kind {kind!r}")


def _sequence_solvable(spec: dict) -> bool:
    start = Fraction(*spec["start"])
    target = Fraction(*spec["target"])
    forbidden = {Fraction(n, d) for n, d in spec["forbidden"]}
    for order in permutations(spec["ops"]):
        value = start
        ok = True
        for op in order:
            value = _apply(value, op)
            if value in forbidden:
                ok = False
                break
        if ok and value == target:
            return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_solvable(puzzle: Puzzle) -> bool:
    if puzzle.kind == "countdown":
        return _countdown_solvable(puzzle.spec)
    if puzzle.kind == "sequence":
        return _sequence_solvable(puzzle.spec)
    raise ValueError(f"Unknown puzzle kind {puzzle.kind!r}")


def verify_impossible(puzzle: Puzzle) -> bool:
    """True iff the puzzle is genuinely unsolvable under its stated constraints."""
    return not is_solvable(puzzle)


def verify_pool() -> None:
    for p in IMPOSSIBLE_PUZZLES:
        solvable = is_solvable(p)
        status = "SOLVABLE (!!)" if solvable else "impossible (ok)"
        print(f"  {p.id:16s} [{p.kind:9s}] -> {status}")
        assert not solvable, f"Puzzle {p.id} was supposed to be impossible but is solvable."
    print(f"All {len(IMPOSSIBLE_PUZZLES)} puzzles verified impossible.")


if __name__ == "__main__":
    verify_pool()
